"""
VisionToll One-Time Final Test Set Evaluation
Academic Context: 23CSE473 Neural Networks and Deep Learning (Group A12)
Evaluates the FROZEN champion model:
  Model: YOLOv8s (Experiment 3)
  Checkpoint: D:/VisionToll/models/exp3_yolov8s/best.pt
  SHA-256: 5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD
Split: TEST (1,083 held-out unseen images, 3,926 GT instances)
Outputs to:
  - results/final_test/
  - results/hard_examples/final_test/
"""

import sys
import json
import time
import csv
import hashlib
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

ROOT          = Path("D:/VisionToll")
DATA_YAML     = ROOT / "data" / "processed" / "vision_toll_yolo" / "data.yaml"
FROZEN_MODEL  = ROOT / "models" / "exp3_yolov8s" / "best.pt"
EXPECTED_HASH = "5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD"

OUT_DIR       = ROOT / "results" / "final_test"
HARD_EX_DIR   = ROOT / "results" / "hard_examples" / "final_test"

OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    print("VISIONTOLL: ONE-TIME FINAL HELD-OUT TEST EVALUATION")
    print("=" * 80)

    # 1. Verify Checkpoint Hash
    print(f"Target Frozen Checkpoint: {FROZEN_MODEL}")
    assert FROZEN_MODEL.exists(), f"Frozen model missing: {FROZEN_MODEL}"

    with open(FROZEN_MODEL, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"Actual SHA-256   : {actual_hash}")
    print(f"Expected SHA-256 : {EXPECTED_HASH}")
    assert actual_hash == EXPECTED_HASH, "CRITICAL ERROR: Checkpoint SHA-256 mismatch!"
    print("Integrity Check Passed: Hash verified.\n")

    # 2. Run Formal Ultralytics Validation on TEST Split
    print("Executing Ultralytics evaluation on TEST split...")
    eval_model = YOLO(str(FROZEN_MODEL))
    
    t0 = time.time()
    val_res = eval_model.val(
        data=str(DATA_YAML),
        split="test",
        batch=16,
        imgsz=640,
        device=0,
        plots=True,
        project=str(ROOT / "results"),
        name="final_test",
        exist_ok=True,
        workers=0
    )
    test_eval_time = time.time() - t0

    mp    = float(val_res.box.mp)
    mr    = float(val_res.box.mr)
    map50 = float(val_res.box.map50)
    map   = float(val_res.box.map)
    f1    = 2 * mp * mr / max(mp + mr, 1e-6)

    speed = val_res.speed
    preprocess_ms = speed.get('preprocess', 0.0)
    inference_ms  = speed.get('inference', 0.0)
    loss_ms       = speed.get('loss', 0.0)
    postprocess_ms= speed.get('postprocess', 0.0)
    total_ms      = preprocess_ms + inference_ms + postprocess_ms
    fps           = 1000.0 / total_ms if total_ms > 0 else 0.0

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION METRICS (OVERALL)")
    print("=" * 80)
    print(f"Precision          : {mp:.4f}")
    print(f"Recall             : {mr:.4f}")
    print(f"F1-Score           : {f1:.4f}")
    print(f"mAP@0.50           : {map50:.4f}")
    print(f"mAP@0.50:0.95      : {map:.4f}")
    print(f"Latency per Image  : {inference_ms:.2f} ms inference ({total_ms:.2f} ms total pipeline)")
    print(f"Throughput         : {fps:.1f} FPS")

    # 3. Per-Class Metrics & CSV
    test_lbl_dir = ROOT / "data" / "processed" / "vision_toll_yolo" / "labels" / "test"
    gt_class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for lp in test_lbl_dir.glob("*.txt"):
        for line in lp.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid in gt_class_counts:
                    gt_class_counts[cid] += 1

    per_class_data = []
    print("\n" + "=" * 80)
    print("PER-CLASS TEST RESULTS")
    print("=" * 80)
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        p_c = float(val_res.box.p[cid]) if cid < len(val_res.box.p) else 0.0
        r_c = float(val_res.box.r[cid]) if cid < len(val_res.box.r) else 0.0
        f1_c = 2 * (p_c * r_c) / max(p_c + r_c, 1e-6)
        m50_c = float(val_res.box.all_ap[cid, 0]) if cid < val_res.box.all_ap.shape[0] else 0.0
        m5095_c = float(val_res.box.maps[cid]) if cid < len(val_res.box.maps) else 0.0
        gt_cnt = gt_class_counts[cid]

        per_class_data.append({
            "Class_ID": cid,
            "Class_Name": cname,
            "GT_Instances": gt_cnt,
            "Precision": round(p_c, 4),
            "Recall": round(r_c, 4),
            "F1_Score": round(f1_c, 4),
            "mAP50": round(m50_c, 4),
            "mAP50_95": round(m5095_c, 4)
        })
        print(f"  Class {cid} ({cname:15s}): GT={gt_cnt:4d} | P={p_c:.4f} | R={r_c:.4f} | F1={f1_c:.4f} | mAP50={m50_c:.4f} | mAP50-95={m5095_c:.4f}")

    # Write per_class_results.csv
    csv_path = OUT_DIR / "per_class_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Class_ID", "Class_Name", "GT_Instances", "Precision", "Recall", "F1_Score", "mAP50", "mAP50_95"])
        w.writeheader()
        w.writerows(per_class_data)
    print(f"\nSaved: {csv_path}")

    # 4. Hard-Example Analysis on Test Split
    print("\n--- Executing Hard-Example Profiling on Test Split ---")
    test_img_dir = ROOT / "data" / "processed" / "vision_toll_yolo" / "images" / "test"
    test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))

    hard_cases_test = []
    low_conf_scenes = 0
    crowded_scenes = 0
    small_obj_scenes = 0
    count_disc_scenes = 0
    complete_miss_scenes = 0

    total_pred_boxes = 0
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

    for idx, img_p in enumerate(test_images, 1):
        lbl_p = test_lbl_dir / (img_p.stem + ".txt")
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

        total_pred_boxes += len(pred_boxes)
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
            hard_cases_test.append({
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

        # Motorcycle specific tracking
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

    print(f"Test Hard-Example Scenes Identified: {len(hard_cases_test)} / {len(test_images)} ({len(hard_cases_test)/len(test_images)*100:.2f}%)")
    
    # Save manifest
    manifest_path = HARD_EX_DIR / "final_test_hard_examples_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(hard_cases_test, f, indent=2)
    print(f"Saved: {manifest_path}")

    # Summary JSON
    summary_data = {
        "evaluation": "VisionToll Final One-Time Held-Out Test Evaluation",
        "frozen_model": str(FROZEN_MODEL),
        "sha256": actual_hash,
        "split": "test",
        "num_test_images": len(test_images),
        "num_gt_instances": sum(gt_class_counts.values()),
        "num_predicted_detections": total_pred_boxes,
        "latency_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total_pipeline": round(total_ms, 2),
            "fps": round(fps, 1)
        },
        "overall_metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map, 4)
        },
        "per_class_metrics": {item["Class_Name"]: item for item in per_class_data},
        "hard_examples": {
            "total_scenes": len(hard_cases_test),
            "low_confidence_scenes": low_conf_scenes,
            "crowded_scenes": crowded_scenes,
            "small_object_scenes": small_obj_scenes,
            "count_discrepancy_scenes": count_disc_scenes,
            "complete_miss_scenes": complete_miss_scenes
        },
        "motorcycle_diagnostics": {
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

    test_res_json = OUT_DIR / "final_test_results.json"
    with open(test_res_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved: {test_res_json}")

    hard_sum_json = HARD_EX_DIR / "final_test_hard_example_summary.json"
    with open(hard_sum_json, "w", encoding="utf-8") as f:
        json.dump(summary_data["hard_examples"], f, indent=2)
    print(f"Saved: {hard_sum_json}")

    print("\n" + "=" * 80)
    print("ONE-TIME FINAL TEST EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
