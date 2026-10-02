"""
VisionToll: Experiment 1 — YOLOv8n Baseline Training Script
Academic Project: 23CSE473 Neural Networks and Deep Learning (Group A12)
Trains YOLOv8n on the 5-class VisionToll dataset for 50 epochs.
Evaluates strictly on the VALIDATION split.
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_YAML = ROOT / "data" / "processed" / "vision_toll_yolo" / "data.yaml"
OUTPUT_DIR = ROOT / "results" / "experiments"
MODEL_SAVE_DIR = ROOT / "models" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

def train_baseline():
    print("=" * 80)
    print("VISIONTOLL: EXPERIMENT 1 — YOLOV8N BASELINE TRAINING")
    print("=" * 80)
    
    # Telemetry
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    print(f"Compute Device   : {device_name}")
    print(f"Dataset Config   : {DATA_YAML}")
    print(f"Target Classes   : {CLASS_NAMES}\n")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at: {DATA_YAML}")

    # Initialize model with pretrained COCO weights
    pretrained_weights = ROOT / "models" / "baseline" / "yolov8n.pt"
    if pretrained_weights.exists():
        model_source = str(pretrained_weights)
    else:
        model_source = "yolov8n.pt"

    print(f"Initializing YOLOv8n from: {model_source}")
    model = YOLO(model_source)

    start_time = time.time()

    # Train for 50 epochs with strict baseline parameters
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=15,
        seed=42,
        project=str(OUTPUT_DIR),
        name="vision_toll_exp1_yolov8n_baseline",
        pretrained=True,
        workers=4,
        plots=True,
        save=True
    )

    duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED in {duration / 60:.2f} minutes ({duration:.1f} seconds)")
    print("=" * 80)

    # Best model path
    run_dir = OUTPUT_DIR / "vision_toll_exp1_yolov8n_baseline"
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        # Copy best model to models/baseline/best.pt (without overwriting yolov8n.pt!)
        dest_best = MODEL_SAVE_DIR / "best.pt"
        import shutil
        shutil.copy2(best_pt, dest_best)
        print(f"Copied best model to: {dest_best}")

    # Formal validation evaluation
    print("\nExecuting formal validation evaluation...")
    eval_model = YOLO(str(best_pt))
    val_metrics = eval_model.val(
        data=str(DATA_YAML),
        split="val",
        plots=True
    )

    mp = float(val_metrics.box.mp)
    mr = float(val_metrics.box.mr)
    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    f1 = 2 * (mp * mr) / max(mp + mr, 1e-6)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: BASELINE VALIDATION RESULTS")
    print("=" * 80)
    print(f"Precision      : {mp:.4f}")
    print(f"Recall         : {mr:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"mAP@0.50       : {map50:.4f}")
    print(f"mAP@0.50:0.95  : {map50_95:.4f}")

    print("\nPer-Class Breakdown:")
    per_class_results = {}
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        p_c = float(val_metrics.box.p[cid]) if cid < len(val_metrics.box.p) else 0.0
        r_c = float(val_metrics.box.r[cid]) if cid < len(val_metrics.box.r) else 0.0
        f1_c = 2 * (p_c * r_c) / max(p_c + r_c, 1e-6)
        map50_c = float(val_metrics.box.maps[cid]) if cid < len(val_metrics.box.maps) else 0.0
        per_class_results[cname] = {
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f1_c, 4),
            "map50": round(map50_c, 4)
        }
        print(f"  Class {cid} ({cname:15s}): P={p_c:.4f} | R={r_c:.4f} | F1={f1_c:.4f} | mAP@0.50={map50_c:.4f}")

    # Save structured results
    report = {
        "experiment": "Experiment 1 — YOLOv8n Baseline",
        "device": device_name,
        "epochs": 50,
        "batch_size": 16,
        "imgsz": 640,
        "seed": 42,
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4)
        },
        "per_class": per_class_results
    }

    report_path = run_dir / "baseline_validation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report to: {report_path}")

    # Mirror artifacts to results/baseline/
    baseline_res_dir = ROOT / "results" / "baseline"
    baseline_res_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for item in run_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, baseline_res_dir / item.name)
        elif item.is_dir() and item.name == "weights":
            (baseline_res_dir / "weights").mkdir(exist_ok=True)
            for w in item.glob("*.pt"):
                shutil.copy2(w, baseline_res_dir / "weights" / w.name)
    shutil.copy2(report_path, baseline_res_dir / "baseline_validation_results.json")
    print(f"Mirrored baseline artifacts to: {baseline_res_dir}")

    # Also log to results/experiments/experiments.csv
    import pandas as pd
    exp_csv = OUTPUT_DIR / "experiments.csv"
    headers = [
        "Experiment_ID", "Model", "Intervention", "Precision", "Recall",
        "F1", "mAP50", "mAP50-95", "Training_Time_min", "Key_Observation"
    ]
    row = {
        "Experiment_ID": "Exp 1",
        "Model": "YOLOv8n",
        "Intervention": "Default baseline",
        "Precision": round(mp, 4),
        "Recall": round(mr, 4),
        "F1": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "Training_Time_min": round(duration / 60, 2),
        "Key_Observation": "Initial baseline; solid Car/Truck detection, baseline for error discovery"
    }
    if exp_csv.exists():
        df = pd.read_csv(exp_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=headers)
    df.to_csv(exp_csv, index=False)
    print(f"Logged experiment record to: {exp_csv}")

if __name__ == '__main__':
    train_baseline()
