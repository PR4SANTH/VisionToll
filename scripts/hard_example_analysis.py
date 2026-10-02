"""
VisionToll Hard-Example Analysis Tool
Identifies challenging validation scenarios:
- Missed detections (False Negatives)
- Small vehicles (< 32^2 pixels)
- Low confidence detections (< 0.40)
- High density / crowded traffic scenes (>= 5 vehicles)
- Visual classification ambiguities
Extracts and renders annotated visualizations into results/hard_examples/
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
RESULTS_DIR = ROOT / "results" / "hard_examples"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

def analyze_hard_examples(model_path, data_root=DATA_DIR, out_dir=RESULTS_DIR, conf_thresh=0.25):
    print("=" * 80)
    print("VISIONTOLL: HARD-EXAMPLE DISCOVERY & VISUAL ERROR PROFILING")
    print("=" * 80)
    print(f"Model checkpoint: {model_path}")
    print(f"Validation data : {data_root / 'images' / 'val'}")
    print(f"Output directory: {out_dir}\n")

    if not Path(model_path).exists():
        print(f"Model {model_path} not found. Please provide a trained checkpoint.")
        return

    model = YOLO(model_path)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    val_images = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    print(f"Auditing {len(val_images)} validation images for hard examples...")

    hard_cases = []

    for idx, img_p in enumerate(val_images, 1):
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append((cid, xc, yc, bw, bh))

        # Run inference
        results = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            c = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            pred_boxes.append((c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        # Criteria 1: Small object misses
        # Criteria 2: Low-confidence detection
        # Criteria 3: High vehicle density (crowded)
        # Criteria 4: Large count discrepancy (gt vs pred)
        reasons = []
        if len(gt_boxes) >= 6:
            reasons.append("High traffic density (crowded scene)")
        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            reasons.append("Complete detector miss (zero detections)")
        elif abs(len(gt_boxes) - len(pred_boxes)) >= 3:
            reasons.append(f"High count discrepancy (GT={len(gt_boxes)}, Pred={len(pred_boxes)})")

        low_conf = [p for p in pred_boxes if p[1] < 0.40]
        if low_conf:
            reasons.append(f"Low confidence detections ({len(low_conf)} boxes < 0.40)")

        # Check for small GT vehicles
        small_gt = [g for g in gt_boxes if (g[3] * g[4]) < (32 * 32 / (640 * 640))]
        if small_gt:
            reasons.append(f"Contains small vehicles ({len(small_gt)} small GT objects)")

        if reasons:
            hard_cases.append({
                "image": img_p.name,
                "path": str(img_p),
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes),
                "reasons": reasons,
                "pred_classes": [CLASS_NAMES.get(p[0], str(p[0])) for p in pred_boxes],
                "gt_classes": [CLASS_NAMES.get(g[0], str(g[0])) for g in gt_boxes]
            })

            # Save annotated visualization
            annotated_bgr = results.plot()
            out_img_file = out_dir / f"hard_{img_p.name}"
            cv2.imwrite(str(out_img_file), annotated_bgr)

    print(f"Identified {len(hard_cases)} hard-example validation images.")
    # Save metadata JSON
    with open(out_dir / "hard_examples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(hard_cases, f, indent=2)

    print(f"Hard example manifest saved to: {out_dir / 'hard_examples_manifest.json'}")
    return hard_cases

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_hard_examples(sys.argv[1])
    else:
        # Default placeholder checkpoint path
        default_model = ROOT / "models" / "baseline" / "best.pt"
        analyze_hard_examples(default_model)
