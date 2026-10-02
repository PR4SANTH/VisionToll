"""
VisionToll Final Unseen Test Set Evaluation
Runs the finalized model strictly once on the untouched TEST set.
Generates:
- Precision, Recall, F1, mAP@0.50, mAP@0.50:0.95
- Per-class metrics for Bus, Car, Motorcycle, Auto Rickshaw, Truck
- Confusion matrices
- Final evaluation plots
"""

import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_YAML = ROOT / "data" / "processed" / "vision_toll_yolo" / "data.yaml"
FINAL_MODEL = ROOT / "models" / "final" / "best.pt"
TEST_RESULTS_DIR = ROOT / "results" / "test"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Bus", "Car", "Motorcycle", "Auto Rickshaw", "Truck"]

def run_final_test(model_path=FINAL_MODEL, data_yaml=DATA_YAML, out_dir=TEST_RESULTS_DIR):
    print("=" * 80)
    print("VISIONTOLL: FINAL UNSEEN TEST EVALUATION")
    print("=" * 80)
    print(f"Final Model Checkpoint: {model_path}")
    print(f"Dataset Configuration : {data_yaml}")
    print(f"Output Directory      : {out_dir}\n")

    if not Path(model_path).exists():
        print(f"Error: Final model {model_path} not found! Please finalize model selection first.")
        return

    model = YOLO(str(model_path))

    print("Executing final evaluation on untouched TEST split...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(out_dir.parent),
        name="test",
        plots=True
    )

    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    f1 = 2 * (mp * mr) / max(mp + mr, 1e-6)

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS (ACADEMIC BENCHMARK)")
    print("=" * 80)
    print(f"Precision      : {mp:.4f}")
    print(f"Recall         : {mr:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"mAP@0.50       : {map50:.4f}")
    print(f"mAP@0.50:0.95  : {map50_95:.4f}")

    print("\nPer-Class Breakdown:")
    per_class_results = {}
    for i, cname in enumerate(CLASS_NAMES):
        c_map50 = metrics.box.maps[i] if i < len(metrics.box.maps) else 0.0
        per_class_results[cname] = {
            "mAP50-95": round(float(c_map50), 4)
        }
        print(f"  {cname:15s}: mAP@0.50:0.95 = {c_map50:.4f}")

    results_data = {
        "model": str(model_path),
        "split": "test",
        "overall": {
            "precision": round(float(mp), 4),
            "recall": round(float(mr), 4),
            "f1": round(float(f1), 4),
            "map50": round(float(map50), 4),
            "map50_95": round(float(map50_95), 4)
        },
        "per_class": per_class_results
    }

    with open(out_dir / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nSaved final test results to: {out_dir / 'final_test_results.json'}")
    return results_data

if __name__ == '__main__':
    run_final_test()
