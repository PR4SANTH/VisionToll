"""
VisionToll Model Comparison & Statistical Evaluation
Compiles validation metrics across:
- Experiment 1: YOLOv8n Baseline
- Experiment 2: YOLOv8n Targeted Improvement
- Experiment 3: YOLOv8s Model Capacity
Outputs structured validation comparison table and saves to results/experiments/
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
RESULTS_DIR = ROOT / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compare_experiments():
    print("=" * 80)
    print("VISIONTOLL: EXPERIMENTAL COMPARISON MATRIX (VALIDATION SET)")
    print("=" * 80)

    csv_path = RESULTS_DIR / "experiments.csv"
    if not csv_path.exists():
        print("Note: experiments.csv does not exist yet. It will be populated as training finishes.")
        # Create initial schema
        headers = [
            "Experiment_ID", "Model", "Intervention", "Precision", "Recall",
            "F1", "mAP50", "mAP50-95", "Training_Time_min", "Key_Observation"
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_path, index=False)
        print(f"Created template at: {csv_path}")
        return df

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("No completed experiments logged yet.")
    else:
        print(df.to_markdown(index=False))

    return df

if __name__ == '__main__':
    compare_experiments()
