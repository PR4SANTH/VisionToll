"""
VisionToll Experiment 2 Hard-Example Evaluation & Motorcycle Failure Profiling
Academic Context: 23CSE473 Neural Networks and Deep Learning (Group A12)
Evaluates Experiment 2 (YOLOv8n + Scale Augmentation) against Experiment 1 Baseline
Strictly evaluates on VALIDATION split (data/processed/vision_toll_yolo/images/val).
Outputs to: results/hard_examples/exp2/
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
VAL_IMG_DIR = DATA_DIR / "images" / "val"
VAL_LBL_DIR = DATA_DIR / "labels" / "val"

EXP1_WEIGHTS = ROOT / "results" / "experiments" / "vision_toll_exp1_yolov8n_baseline" / "weights" / "best.pt"
EXP2_WEIGHTS = ROOT / "results" / "experiments" / "vision_toll_exp2_scale_augmentation" / "weights" / "best.pt"

OUT_DIR_EXP2 = ROOT / "results" / "hard_examples" / "exp2"
OUT_DIR_EXP2.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

def box_iou_xywhn(b1, b2):
    # b1, b2 are [xc, yc, w, h] normalized
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

def run_evaluation():
    print("=" * 80)
    print("VISIONTOLL: EXPERIMENT 2 HARD-EXAMPLE EVALUATION")
    print("=" * 80)
    print(f"Exp2 Checkpoint : {EXP2_WEIGHTS}")
    print(f"Validation Set  : {VAL_IMG_DIR}")
    print(f"Output Directory: {OUT_DIR_EXP2}\n")

    assert EXP2_WEIGHTS.exists(), f"Exp2 weights missing: {EXP2_WEIGHTS}"
    assert VAL_IMG_DIR.exists(), f"Val images missing: {VAL_IMG_DIR}"

    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png")))
    print(f"Found {len(val_images)} validation images to evaluate.")

    # Load Exp2 model
    model2 = YOLO(str(EXP2_WEIGHTS))
    
    # 1. Run Exp2 Hard Example Discovery (exact Exp1 criteria)
    conf_thresh = 0.25
    hard_cases_exp2 = []
    
    # Detailed category accumulators
    low_conf_scenes = 0
    crowded_scenes = 0
    small_obj_scenes = 0
    count_disc_scenes = 0
    complete_miss_scenes = 0

    # Motorcycle specific metrics for Exp2
    m_gt_total = 0
    m_pred_total = 0
    m_tp_total = 0 # matched IoU >= 0.5
    m_fn_total = 0 # missed
    m_fp_total = 0 # unmatched pred
    m_conf_list = []
    m_small_gt_total = 0
    m_small_gt_detected = 0
    m_crowded_gt_total = 0
    m_crowded_gt_detected = 0
    m_low_conf_count = 0

    print("Running Exp2 inference across validation split...")
    t0 = time.time()

    for idx, img_p in enumerate(val_images, 1):
        lbl_p = VAL_LBL_DIR / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append((cid, xc, yc, bw, bh))

        # Predict with Exp2
        res2 = model2.predict(source=str(img_p), conf=conf_thresh, verbose=False)[0]
        pred_boxes = []
        for box in res2.boxes:
            c = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            pred_boxes.append((c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        # Original Criteria:
        # Criteria 1: Crowded scene (>=6)
        # Criteria 2: Zero detections
        # Criteria 3: High count discrepancy (>=3)
        # Criteria 4: Low confidence detections (< 0.40)
        # Criteria 5: Small GT vehicles (< 32^2 px)
        reasons = []
        is_crowded = len(gt_boxes) >= 6
        if is_crowded:
            crowded_scenes += 1
            reasons.append("High traffic density (crowded scene)")
            
        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            complete_miss_scenes += 1
            reasons.append("Complete detector miss (zero detections)")
        elif abs(len(gt_boxes) - len(pred_boxes)) >= 3:
            count_disc_scenes += 1
            reasons.append(f"High count discrepancy (GT={len(gt_boxes)}, Pred={len(pred_boxes)})")

        low_conf = [p for p in pred_boxes if p[1] < 0.40]
        if low_conf:
            low_conf_scenes += 1
            reasons.append(f"Low confidence detections ({len(low_conf)} boxes < 0.40)")

        small_gt = [g for g in gt_boxes if (g[3] * g[4]) < (32 * 32 / (640 * 640))]
        if small_gt:
            small_obj_scenes += 1
            reasons.append(f"Contains small vehicles ({len(small_gt)} small GT objects)")

        if reasons:
            hard_cases_exp2.append({
                "image": img_p.name,
                "path": str(img_p),
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes),
                "reasons": reasons,
                "pred_classes": [CLASS_NAMES.get(p[0], str(p[0])) for p in pred_boxes],
                "gt_classes": [CLASS_NAMES.get(g[0], str(g[0])) for g in gt_boxes]
            })

            # Save annotated visualization for hard cases
            annotated_bgr = res2.plot()
            out_img_file = OUT_DIR_EXP2 / f"hard_{img_p.name}"
            cv2.imwrite(str(out_img_file), annotated_bgr)

        # Motorcycle specific tracking
        gt_moto = [g for g in gt_boxes if g[0] == 2]
        pred_moto = [p for p in pred_boxes if p[0] == 2]

        m_gt_total += len(gt_moto)
        m_pred_total += len(pred_moto)

        for pm in pred_moto:
            m_conf_list.append(pm[1])
            if pm[1] < 0.40:
                m_low_conf_count += 1

        # Match motorcycle GT to predictions
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

    duration = time.time() - t0
    print(f"Exp2 validation inference complete in {duration:.2f} s ({duration/len(val_images)*1000:.1f} ms/image).")
    print(f"Exp2 Hard-Example Scenes Identified: {len(hard_cases_exp2)}")

    # Save manifest
    manifest_path = OUT_DIR_EXP2 / "exp2_hard_examples_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(hard_cases_exp2, f, indent=2)
    print(f"Exp2 Manifest saved: {manifest_path}")

    # Now let's calculate the exact matched metrics for Exp1 using model1
    print("\nRunning matching evaluation for Exp1 baseline checkpoint...")
    model1 = YOLO(str(EXP1_WEIGHTS))
    
    m1_gt_total = 0
    m1_pred_total = 0
    m1_tp_total = 0
    m1_fn_total = 0
    m1_fp_total = 0
    m1_conf_list = []
    m1_small_gt_total = 0
    m1_small_gt_detected = 0
    m1_crowded_gt_total = 0
    m1_crowded_gt_detected = 0
    m1_low_conf_count = 0

    for idx, img_p in enumerate(val_images, 1):
        lbl_p = VAL_LBL_DIR / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append((cid, xc, yc, bw, bh))

        res1 = model1.predict(source=str(img_p), conf=conf_thresh, verbose=False)[0]
        pred_boxes1 = []
        for box in res1.boxes:
            c = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            pred_boxes1.append((c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        is_crowded = len(gt_boxes) >= 6
        gt_moto1 = [g for g in gt_boxes if g[0] == 2]
        pred_moto1 = [p for p in pred_boxes1 if p[0] == 2]

        m1_gt_total += len(gt_moto1)
        m1_pred_total += len(pred_moto1)

        for pm in pred_moto1:
            m1_conf_list.append(pm[1])
            if pm[1] < 0.40:
                m1_low_conf_count += 1

        matched_gt1 = set()
        matched_pred1 = set()
        for g_idx, g in enumerate(gt_moto1):
            is_small = (g[3] * g[4]) < (32 * 32 / (640 * 640))
            if is_small:
                m1_small_gt_total += 1
            if is_crowded:
                m1_crowded_gt_total += 1

            best_iou = 0.0
            best_p_idx = -1
            for p_idx, p in enumerate(pred_moto1):
                if p_idx in matched_pred1:
                    continue
                iou = box_iou_xywhn(g[1:], p[2:])
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
            
            if best_iou >= 0.50:
                matched_gt1.add(g_idx)
                matched_pred1.add(best_p_idx)
                if is_small:
                    m1_small_gt_detected += 1
                if is_crowded:
                    m1_crowded_gt_detected += 1

        m1_tp_total += len(matched_gt1)
        m1_fn_total += (len(gt_moto1) - len(matched_gt1))
        m1_fp_total += (len(pred_moto1) - len(matched_pred1))

    # Summary dictionary
    summary = {
        "evaluation": "VisionToll Experiment 2 Hard-Example Evaluation",
        "checkpoint": str(EXP2_WEIGHTS),
        "split": "val",
        "num_val_images": len(val_images),
        "hard_examples_count": len(hard_cases_exp2),
        "categories": {
            "low_confidence_scenes": low_conf_scenes,
            "crowded_scenes": crowded_scenes,
            "small_object_scenes": small_obj_scenes,
            "count_discrepancy_scenes": count_disc_scenes,
            "complete_miss_scenes": complete_miss_scenes
        },
        "motorcycle_metrics": {
            "exp1": {
                "gt_total": m1_gt_total,
                "pred_total": m1_pred_total,
                "matched_tp": m1_tp_total,
                "missed_fn": m1_fn_total,
                "false_positives": m1_fp_total,
                "recall": round(m1_tp_total / max(m1_gt_total, 1), 4),
                "precision": round(m1_tp_total / max(m1_pred_total, 1), 4),
                "small_gt_total": m1_small_gt_total,
                "small_gt_detected": m1_small_gt_detected,
                "small_gt_missed": m1_small_gt_total - m1_small_gt_detected,
                "small_recall": round(m1_small_gt_detected / max(m1_small_gt_total, 1), 4),
                "crowded_gt_total": m1_crowded_gt_total,
                "crowded_gt_detected": m1_crowded_gt_detected,
                "crowded_gt_missed": m1_crowded_gt_total - m1_crowded_gt_detected,
                "crowded_recall": round(m1_crowded_gt_detected / max(m1_crowded_gt_total, 1), 4),
                "low_conf_preds": m1_low_conf_count,
                "mean_conf": round(float(np.mean(m1_conf_list)), 4) if m1_conf_list else 0.0,
                "median_conf": round(float(np.median(m1_conf_list)), 4) if m1_conf_list else 0.0
            },
            "exp2": {
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
    }

    sum_path = OUT_DIR_EXP2 / "exp2_hard_example_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved: {sum_path}")

    return summary

if __name__ == '__main__':
    run_evaluation()
