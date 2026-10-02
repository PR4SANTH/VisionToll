import json
import csv
from pathlib import Path

p = Path("d:/VisionToll/results/diagnostics/visiontollnet_stage1_forensics")
with open(p / "stage1_forensics.json", encoding="utf-8") as f:
    data = json.load(f)

pred_stats = data["prediction_statistics"]
with open(p / "prediction_stats.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Checkpoint", "Level", "Logit_Min", "Logit_Max", "Logit_Mean", "Prob_Mean", "Pct_Gt_0_05", "Box_W_Mean", "Box_H_Mean"])
    for ckpt, lvls in pred_stats.items():
        for lvl, s in lvls.items():
            w.writerow([
                ckpt, lvl,
                f"{s['logit_min']:.2f}",
                f"{s['logit_max']:.2f}",
                f"{s['logit_mean']:.2f}",
                f"{s['prob_mean']:.4f}",
                f"{s['pct_prob_gt_0_05']:.2f}%",
                f"{s['box_w_mean']:.2f}",
                f"{s['box_h_mean']:.2f}"
            ])
print("prediction_stats.csv created successfully")
