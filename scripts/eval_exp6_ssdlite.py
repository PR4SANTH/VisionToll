import os
import sys
import time
import json
import csv
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from exp6_ssdlite_utils import VisionTollSSDLiteDataset, load_checkpoint
from torchmetrics.detection.mean_ap import MeanAveragePrecision

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
EXP_DIR = ROOT / "results" / "experiments" / "vision_toll_exp6_ssdlite"
HARD_DIR = EXP_DIR / "hard_examples"
HARD_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {1: "Bus", 2: "Car", 3: "Motorcycle", 4: "Auto Rickshaw", 5: "Truck"}
NUM_CLASSES = 6

def compute_tp_fp_fn(preds, targets, iou_thresh=0.5, conf_thresh=0.25):
    # Simplified matching for a fixed confidence to get proxy P, R, F1
    tp_per_class = {c: 0 for c in range(1, 6)}
    fp_per_class = {c: 0 for c in range(1, 6)}
    fn_per_class = {c: 0 for c in range(1, 6)}
    
    for p, t in zip(preds, targets):
        p_boxes = p['boxes']
        p_scores = p['scores']
        p_labels = p['labels']
        t_boxes = t['boxes']
        t_labels = t['labels']
        
        keep = p_scores >= conf_thresh
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        p_labels = p_labels[keep]
        
        from torchvision.ops import box_iou
        if len(p_boxes) > 0 and len(t_boxes) > 0:
            ious = box_iou(p_boxes, t_boxes)
        else:
            ious = torch.zeros((len(p_boxes), len(t_boxes)))
            
        t_matched = {i: False for i in range(len(t_boxes))}
        
        # Sort predictions by score
        if len(p_boxes) > 0:
            sorted_indices = torch.argsort(p_scores, descending=True)
            for p_idx in sorted_indices:
                pl = p_labels[p_idx].item()
                
                best_iou = 0
                best_t_idx = -1
                for t_idx in range(len(t_boxes)):
                    tl = t_labels[t_idx].item()
                    if pl == tl and not t_matched[t_idx]:
                        iou = ious[p_idx, t_idx].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_t_idx = t_idx
                
                if best_iou >= iou_thresh:
                    t_matched[best_t_idx] = True
                    tp_per_class[pl] += 1
                else:
                    fp_per_class[pl] += 1
                    
        for t_idx in range(len(t_boxes)):
            if not t_matched[t_idx]:
                tl = t_labels[t_idx].item()
                fn_per_class[tl] += 1
                
    return tp_per_class, fp_per_class, fn_per_class

def draw_boxes(img_path, p_boxes, p_labels, p_scores, t_boxes, t_labels, conf_thresh=0.25):
    img = cv2.imread(str(img_path))
    # Draw GT in green
    for box, lbl in zip(t_boxes, t_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{CLASS_NAMES[lbl.item()]}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Draw Preds in red
    for box, lbl, score in zip(p_boxes, p_labels, p_scores):
        if score.item() >= conf_thresh:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{CLASS_NAMES[lbl.item()]}:{score.item():.2f}", (x2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    return img

def main():
    print("=" * 80)
    print("EXPERIMENT 6 EVALUATION & REPORTING")
    print("=" * 80)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load Model
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
    epoch, metrics = load_checkpoint(EXP_DIR / "weights" / "best.pt", model)
    model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded best.pt (Epoch {epoch}) - Params: {param_count}")
    
    val_dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val", img_size=(320, 320))
    val_dataset_original_size = VisionTollSSDLiteDataset(DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val", img_size=(640, 640)) # for hard examples original aspect context if needed, but we'll stick to 320 for pred

    metric = MeanAveragePrecision(class_metrics=True)
    all_preds = []
    all_targets = []
    
    # Inference Latency
    print("Measuring inference latency...")
    latency_times = []
    with torch.no_grad():
        for i in range(110):
            if i >= len(val_dataset): break
            img, _ = val_dataset[i]
            img = img.to(device).unsqueeze(0)
            
            t0 = time.time()
            _ = model(img)
            t1 = time.time()
            
            if i >= 10: # warmup
                latency_times.append((t1 - t0) * 1000)
                
    avg_latency = sum(latency_times) / len(latency_times) if latency_times else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    print(f"Latency: {avg_latency:.2f} ms ({fps:.2f} FPS)")
    
    # Validation loop
    print("Running validation metrics and hard-example analysis...")
    hard_cases = []
    
    tp_cls, fp_cls, fn_cls = {c: 0 for c in range(1, 6)}, {c: 0 for c in range(1, 6)}, {c: 0 for c in range(1, 6)}
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img, target = val_dataset[i]
            img_tensor = img.to(device).unsqueeze(0)
            
            preds = model(img_tensor)[0]
            
            p = {k: v.cpu() for k, v in preds.items()}
            t = {k: v.cpu() for k, v in target.items()}
            
            metric.update([p], [t])
            all_preds.append(p)
            all_targets.append(t)
            
            # Hard Examples check (similar to YOLO script)
            gt_boxes = t['boxes']
            pred_boxes_conf = p['boxes'][p['scores'] >= 0.25]
            
            reasons = []
            if len(gt_boxes) >= 6:
                reasons.append("High traffic density (crowded scene)")
            if len(gt_boxes) > 0 and len(pred_boxes_conf) == 0:
                reasons.append("Complete detector miss")
            elif abs(len(gt_boxes) - len(pred_boxes_conf)) >= 3:
                reasons.append(f"Count discrepancy (GT={len(gt_boxes)}, Pred={len(pred_boxes_conf)})")
                
            small_gt = [b for b in gt_boxes if ((b[2]-b[0])*(b[3]-b[1])) < (32*32*320*320/(640*640))]
            if small_gt:
                reasons.append("Contains small vehicles")
                
            if reasons:
                img_path = val_dataset.images[i]
                annotated = draw_boxes(img_path, p['boxes'], p['labels'], p['scores'], t['boxes'], t['labels'])
                cv2.imwrite(str(HARD_DIR / f"hard_{img_path.name}"), annotated)
                hard_cases.append({"image": img_path.name, "reasons": reasons})
                
    results = metric.compute()
    map50 = results['map_50'].item()
    map50_95 = results['map'].item()
    map50_per_class = results['map_50_per_class'].tolist() if 'map_50_per_class' in results else [0]*5
    map_per_class = results['map_per_class'].tolist() if 'map_per_class' in results else [0]*5
    
    tp_cls, fp_cls, fn_cls = compute_tp_fp_fn(all_preds, all_targets)
    
    overall_tp = sum(tp_cls.values())
    overall_fp = sum(fp_cls.values())
    overall_fn = sum(fn_cls.values())
    
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    class_metrics = {}
    for c in range(1, 6):
        p_c = tp_cls[c] / (tp_cls[c] + fp_cls[c]) if (tp_cls[c] + fp_cls[c]) > 0 else 0
        r_c = tp_cls[c] / (tp_cls[c] + fn_cls[c]) if (tp_cls[c] + fn_cls[c]) > 0 else 0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0
        
        class_metrics[CLASS_NAMES[c]] = {
            "precision": p_c,
            "recall": r_c,
            "f1": f1_c,
            "map50": map50_per_class[c-1],
            "map50_95": map_per_class[c-1]
        }
        
    print(f"Overall - P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f} mAP50: {map50:.4f} mAP50-95: {map50_95:.4f}")
    
    # Read training stats
    total_time = 0
    peak_vram = 0
    if (EXP_DIR / "progress.csv").exists():
        with open(EXP_DIR / "progress.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_time += float(row['epoch_time_sec'])
                peak_vram = max(peak_vram, float(row['peak_vram_mb']))
                
    train_minutes = total_time / 60
    
    # Write report
    report_content = f"""# Experiment 6: SSD-Lite MobileNetV3 Large

## 1. Objective
Train and evaluate SSD-Lite MobileNetV3 Large as the fifth distinct detector architecture for the VisionToll six-model comparison.

## 2. Architecture & Model
- **Model**: torchvision `ssdlite320_mobilenet_v3_large`
- **Input Size**: 320x320 (native SSD-Lite resolution)
- **Parameters**: {param_count:,}
- **Pretrained Weights**: `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1` (Detection head reinitialized for 5 foreground classes)

## 3. Dataset & Class Mapping
- **Train**: 3,535 images | **Val**: 884 images
- **Classes**: 0=Background, 1=Bus, 2=Car, 3=Motorcycle, 4=Auto Rickshaw, 5=Truck
- **Test Set**: LOCKED - NOT USED

## 4. Training Configuration
- **Epochs**: {epoch}
- **Batch Size**: 4
- **Optimizer**: AdamW (lr=0.001)
- **Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Peak VRAM**: {peak_vram:.2f} MB
- **Training Time**: {train_minutes:.2f} minutes

## 5. Validation Metrics (Overall)
*Note: P/R/F1 evaluated at fixed conf=0.25*
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1**: {f1:.4f}
- **mAP50**: {map50:.4f}
- **mAP50-95**: {map50_95:.4f}

## 6. Per-Class Metrics
| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|-------|-----------|--------|----|-------|----------|
"""
    for name, m in class_metrics.items():
        report_content += f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['map50']:.4f} | {m['map50_95']:.4f} |\n"

    report_content += f"""
## 7. Hard-Example Analysis
- **Hard-example scenes identified**: {len(hard_cases)}
These scenes were extracted to `results/experiments/vision_toll_exp6_ssdlite/hard_examples/` for containing dense traffic, small vehicles, or missing predictions.

## 8. Inference Benchmark
- **Latency (batch=1)**: {avg_latency:.2f} ms
- **FPS**: {fps:.2f}

## 9. Comparison with YOLOv8s (Exp 3)
SSD-Lite achieves mAP50-95 of {map50_95:.4f} vs YOLOv8s {0.7399}. SSD-Lite is significantly lighter (~3.4M params vs YOLOv8s ~11M) and uses a smaller input resolution (320x320), making it highly efficient but generally less accurate than the YOLO series on small objects.
"""
    
    with open(EXP_DIR / "exp6_ssdlite_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report saved to {EXP_DIR / 'exp6_ssdlite_report.md'}")
    
    # Append to experiments.csv
    csv_file = ROOT / "results" / "experiments" / "experiments.csv"
    file_exists = csv_file.exists()
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Experiment", "Architecture", "Precision", "Recall", "F1", "mAP50", "mAP50_95", "Parameters", "Latency_ms"])
        writer.writerow(["Experiment 6", "SSD-Lite MobileNetV3 Large", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{map50:.4f}", f"{map50_95:.4f}", param_count, f"{avg_latency:.2f}"])

    print("=" * 50)
    print("EXPERIMENT 6 — SSD-LITE COMPLETE")
    print("=" * 50)
    print(f"Model:\nSSD-Lite MobileNetV3 Large")
    print(f"Best Epoch:\n{epoch}")
    print(f"Epochs Completed:\n50")
    print(f"Training Time:\n{train_minutes:.2f} min")
    print(f"Parameters:\n{param_count}")
    print(f"Batch Size:\n4")
    print(f"Peak VRAM:\n{peak_vram:.2f} MB")
    print(f"Validation:\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nmAP50: {map50:.4f}\nmAP50-95: {map50_95:.4f}")
    
    mc = class_metrics["Motorcycle"]
    print(f"\nMotorcycle:\nPrecision: {mc['precision']:.4f}\nRecall: {mc['recall']:.4f}\nF1: {mc['f1']:.4f}\nmAP50: {mc['map50']:.4f}\nmAP50-95: {mc['map50_95']:.4f}")
    print(f"\nHard-example scenes:\n{len(hard_cases)}")
    print(f"Inference latency:\n{avg_latency:.2f} ms")
    print(f"Approx FPS:\n{fps:.2f}")
    print(f"Test Set:\nLOCKED — NOT USED")
    print(f"\nNext:\nEXPERIMENT 7 — RETINANET RESNET50-FPN")
    print("=" * 50)

if __name__ == "__main__":
    main()
