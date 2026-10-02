"""
VisionToll Experiment 3: YOLOv8s Model Capacity Comparison
Academic Project: 23CSE473 Neural Networks and Deep Learning (Group A12)
Trains YOLOv8s (11.2M parameters) on the 5-class VisionToll dataset for 50 epochs.
Strictly evaluates on the VALIDATION split.
Includes automated post-training validation and hard-example profiling.
"""

import os
import sys
import time
import json
import shutil
import csv
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT           = Path("D:/VisionToll")
DATA_YAML      = ROOT / "data" / "processed" / "vision_toll_yolo" / "data.yaml"
OUTPUT_DIR     = ROOT / "results" / "experiments"
EXP_NAME       = "vision_toll_exp3_yolov8s"
MODEL_SAVE_DIR = ROOT / "models" / "exp3_yolov8s"
HARD_EX_DIR    = ROOT / "results" / "hard_examples" / "exp3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
HARD_EX_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

def box_iou_xywhn(b1, b2):
    x1_min = b1[0] - b1[2] / 2.0
    y1_min = b1[1] - b1[3] / 2.0
    x1_max = b1[0] + b1[2] / 2.0
    y1_max = b1[1] + b1[3] / 2.0

    x2_min = b2[0] - b2[2] / 2.0
    y2_min = b2[1] - b2[3] / 2.0
    x2_max = b2[0] + b2[2] / 2.0
    y2_max = b2[1] + b2[3] / 2.0

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def main():
    print("=" * 80)
    print("VISIONTOLL: EXPERIMENT 3 — YOLOV8S MODEL CAPACITY COMPARISON")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_ver    = torch.version.cuda or "N/A"
    print(f"Device        : {device_name}")
    print(f"CUDA Version  : {cuda_ver}")
    print(f"PyTorch Ver   : {torch.__version__}")
    print(f"Dataset Config: {DATA_YAML}")
    print(f"Target Classes: {CLASS_NAMES}\n")

    # Step 1: Initialize model
    pretrained_weights = ROOT / "yolov8s.pt"
    if not pretrained_weights.exists():
        model_source = "yolov8s.pt"
    else:
        model_source = str(pretrained_weights)

    print(f"Loading pretrained backbone from: {model_source}")
    model = YOLO(model_source)

    # Step 2: Train
    print("\nStarting 50-epoch training run for YOLOv8s...")
    t0 = time.time()
    
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        seed=42,
        deterministic=True,
        workers=4,
        project=str(OUTPUT_DIR),
        name=EXP_NAME,
        pretrained=True,
        plots=True,
        save=True,
        scale=0.5,
        copy_paste=0.0,
        close_mosaic=10,
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        cutmix=0.0,
        erasing=0.4
    )

    duration = time.time() - t0
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED in {duration / 60:.2f} minutes ({duration:.1f} seconds)")
    print("=" * 80)

    # Step 3: Checkpoints
    run_dir = OUTPUT_DIR / EXP_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    assert best_pt.exists(), f"Missing best checkpoint: {best_pt}"
    shutil.copy2(best_pt, MODEL_SAVE_DIR / "best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, MODEL_SAVE_DIR / "last.pt")
    print(f"Mirrored best weights to: {MODEL_SAVE_DIR / 'best.pt'}")

    # Step 4: Formal Validation Evaluation
    print("\nExecuting formal validation evaluation on best.pt...")
    eval_model = YOLO(str(best_pt))
    val_metrics = eval_model.val(
        data=str(DATA_YAML),
        split="val",
        batch=16,
        imgsz=640,
        device=0,
        plots=True,
        workers=0
    )

    mp    = float(val_metrics.box.mp)
    mr    = float(val_metrics.box.mr)
    map50 = float(val_metrics.box.map50)
    map   = float(val_metrics.box.map)
    f1    = 2 * mp * mr / max(mp + mr, 1e-6)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: YOLOV8S VALIDATION RESULTS")
    print("=" * 80)
    print(f"Precision      : {mp:.4f}")
    print(f"Recall         : {mr:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"mAP@0.50       : {map50:.4f}")
    print(f"mAP@0.50:0.95  : {map:.4f}")

    per_class_results = {}
    print("\nPer-Class Breakdown:")
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        p_c = float(val_metrics.box.p[cid]) if cid < len(val_metrics.box.p) else 0.0
        r_c = float(val_metrics.box.r[cid]) if cid < len(val_metrics.box.r) else 0.0
        f1_c = 2 * (p_c * r_c) / max(p_c + r_c, 1e-6)
        m50_c = float(val_metrics.box.all_ap[cid, 0]) if cid < val_metrics.box.all_ap.shape[0] else 0.0
        m5095_c = float(val_metrics.box.maps[cid]) if cid < len(val_metrics.box.maps) else 0.0
        per_class_results[cname] = {
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f1_c, 4),
            "map50": round(m50_c, 4),
            "map50_95": round(m5095_c, 4)
        }
        print(f"  Class {cid} ({cname:15s}): P={p_c:.4f} | R={r_c:.4f} | F1={f1_c:.4f} | mAP@0.50={m50_c:.4f} | mAP@0.50:0.95={m5095_c:.4f}")

    # Save JSON summary
    report = {
        "experiment": "Experiment 3 — YOLOv8s Model Capacity Comparison",
        "device": device_name,
        "cuda": cuda_ver,
        "pytorch": torch.__version__,
        "epochs": 50,
        "batch_size": 16,
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "seed": 42,
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map, 4)
        },
        "per_class": per_class_results
    }
    with open(run_dir / "exp3_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved: {run_dir / 'exp3_validation_results.json'}")

    # Step 5: Hard-Example Discovery & Motorcycle Profiling
    print("\nExecuting Hard-Example Discovery for Exp3...")
    val_img_dir = ROOT / "data" / "processed" / "vision_toll_yolo" / "images" / "val"
    val_lbl_dir = ROOT / "data" / "processed" / "vision_toll_yolo" / "labels" / "val"
    val_images = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))

    hard_cases_exp3 = []
    low_conf_scenes = 0
    crowded_scenes = 0
    small_obj_scenes = 0
    count_disc_scenes = 0
    complete_miss_scenes = 0

    m_gt_total = 0
    m_pred_total = 0
    m_tp_total = 0
    m_fn_total = 0
    m_fp_total = 0
    m_conf_list = []
    m_small_gt_total = 0
    m_small_gt_detected = 0
    m_crowded_gt_total = 0
    m_crowded_gt_detected = 0
    m_low_conf_count = 0

    for idx, img_p in enumerate(val_images, 1):
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append((cid, xc, yc, bw, bh))

        res = eval_model.predict(source=str(img_p), conf=0.25, verbose=False)[0]
        pred_boxes = []
        for box in res.boxes:
            c = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            pred_boxes.append((c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        is_crowded = len(gt_boxes) >= 6
        if is_crowded:
            crowded_scenes += 1
        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            complete_miss_scenes += 1
        elif abs(len(gt_boxes) - len(pred_boxes)) >= 3:
            count_disc_scenes += 1

        low_conf = [p for p in pred_boxes if p[1] < 0.40]
        if low_conf:
            low_conf_scenes += 1

        small_gt = [g for g in gt_boxes if (g[3] * g[4]) < (32 * 32 / (640 * 640))]
        if small_gt:
            small_obj_scenes += 1

        # Check if hard case
        reasons = []
        if is_crowded:
            reasons.append("High traffic density (crowded scene)")
        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            reasons.append("Complete detector miss (zero detections)")
        elif abs(len(gt_boxes) - len(pred_boxes)) >= 3:
            reasons.append(f"High count discrepancy (GT={len(gt_boxes)}, Pred={len(pred_boxes)})")
        if low_conf:
            reasons.append(f"Low confidence detections ({len(low_conf)} boxes < 0.40)")
        if small_gt:
            reasons.append(f"Contains small vehicles ({len(small_gt)} small GT objects)")

        if reasons:
            hard_cases_exp3.append({
                "image": img_p.name,
                "path": str(img_p),
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes),
                "reasons": reasons,
                "pred_classes": [CLASS_NAMES.get(p[0], str(p[0])) for p in pred_boxes],
                "gt_classes": [CLASS_NAMES.get(g[0], str(g[0])) for g in gt_boxes]
            })
            annotated_bgr = res.plot()
            cv2.imwrite(str(HARD_EX_DIR / f"hard_{img_p.name}"), annotated_bgr)

        # Motorcycle specific
        gt_moto = [g for g in gt_boxes if g[0] == 2]
        pred_moto = [p for p in pred_boxes if p[0] == 2]

        m_gt_total += len(gt_moto)
        m_pred_total += len(pred_moto)

        for pm in pred_moto:
            m_conf_list.append(pm[1])
            if pm[1] < 0.40:
                m_low_conf_count += 1

        matched_gt = set()
        matched_pred = set()
        for g_idx, g in enumerate(gt_moto):
            is_small = (g[3] * g[4]) < (32 * 32 / (640 * 640))
            if is_small:
                m_small_gt_total += 1
            if is_crowded:
                m_crowded_gt_total += 1

            best_iou = 0.0
            best_p_idx = -1
            for p_idx, p in enumerate(pred_moto):
                if p_idx in matched_pred:
                    continue
                iou = box_iou_xywhn(g[1:], p[2:])
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
            
            if best_iou >= 0.50:
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)
                if is_small:
                    m_small_gt_detected += 1
                if is_crowded:
                    m_crowded_gt_detected += 1

        m_tp_total += len(matched_gt)
        m_fn_total += (len(gt_moto) - len(matched_gt))
        m_fp_total += (len(pred_moto) - len(matched_pred))

    # Save manifest & summary
    with open(HARD_EX_DIR / "exp3_hard_examples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(hard_cases_exp3, f, indent=2)

    hard_summary = {
        "evaluation": "VisionToll Experiment 3 Hard-Example Evaluation",
        "checkpoint": str(best_pt),
        "split": "val",
        "num_val_images": len(val_images),
        "hard_examples_count": len(hard_cases_exp3),
        "categories": {
            "low_confidence_scenes": low_conf_scenes,
            "crowded_scenes": crowded_scenes,
            "small_object_scenes": small_obj_scenes,
            "count_discrepancy_scenes": count_disc_scenes,
            "complete_miss_scenes": complete_miss_scenes
        },
        "motorcycle_metrics": {
            "gt_total": m_gt_total,
            "pred_total": m_pred_total,
            "matched_tp": m_tp_total,
            "missed_fn": m_fn_total,
            "false_positives": m_fp_total,
            "recall": round(m_tp_total / max(m_gt_total, 1), 4),
            "precision": round(m_tp_total / max(m_pred_total, 1), 4),
            "small_gt_total": m_small_gt_total,
            "small_gt_detected": m_small_gt_detected,
            "small_gt_missed": m_small_gt_total - m_small_gt_detected,
            "small_recall": round(m_small_gt_detected / max(m_small_gt_total, 1), 4),
            "crowded_gt_total": m_crowded_gt_total,
            "crowded_gt_detected": m_crowded_gt_detected,
            "crowded_gt_missed": m_crowded_gt_total - m_crowded_gt_detected,
            "crowded_recall": round(m_crowded_gt_detected / max(m_crowded_gt_total, 1), 4),
            "low_conf_preds": m_low_conf_count,
            "mean_conf": round(float(np.mean(m_conf_list)), 4) if m_conf_list else 0.0,
            "median_conf": round(float(np.median(m_conf_list)), 4) if m_conf_list else 0.0
        }
    }
    with open(HARD_EX_DIR / "exp3_hard_example_summary.json", "w", encoding="utf-8") as f:
        json.dump(hard_summary, f, indent=2)

    # Step 6: Update experiments.csv
    exp_csv = OUTPUT_DIR / "experiments.csv"
    fn = ['Experiment_ID','Model','Intervention','Precision','Recall','F1','mAP50','mAP50-95','Training_Time_min','Key_Observation']
    nr = {
        'Experiment_ID': 'Exp 3',
        'Model': 'YOLOv8s',
        'Intervention': 'Model Capacity (11.2M params)',
        'Precision': round(mp, 4),
        'Recall': round(mr, 4),
        'F1': round(f1, 4),
        'mAP50': round(map50, 4),
        'mAP50-95': round(map, 4),
        'Training_Time_min': round(duration / 60, 2),
        'Key_Observation': 'Capacity comparison; evaluated on validation split'
    }
    rows = []
    if exp_csv.exists():
        for r in csv.DictReader(open(exp_csv, encoding='utf-8')):
            if r.get('Experiment_ID') != 'Exp 3':
                rows.append(r)
    rows.append(nr)
    with open(exp_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETE — TEST SET QUARANTINED AND UNTOUCHED.")
    print("=" * 80)

if __name__ == '__main__':
    main()
