"""generate_forensic_plots.py

Generate diagnostic visualizations for the VisionTollNet Stage 1 failure forensics.
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("d:/VisionToll")
FORENSICS_DIR = ROOT / "results" / "diagnostics" / "visiontollnet_stage1_forensics"
FORENSICS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def plot_assignment_starvation():
    """Plot positive assignments per level across Stage 0, Stage 1 Best, and Stage 1 Last."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stages = ["Stage 0 (Baseline)", "Stage 1 (Epoch 2 - Best)", "Stage 1 (Epoch 20 - Last)"]
    levels = ["P2", "P3", "P4", "P5"]
    
    # Assignments data from forensics JSON
    data = {
        "P2": [0, 46, 120],
        "P3": [65, 57, 0],
        "P4": [27, 17, 0],
        "P5": [28, 0, 0],
    }
    
    colors = {
        "P2": "#2b5c8f",
        "P3": "#389e82",
        "P4": "#e69f00",
        "P5": "#d55e00",
    }
    
    x = np.arange(len(stages))
    width = 0.18
    multiplier = 0
    
    for lvl in levels:
        offset = width * multiplier
        rects = ax.bar(x + offset, data[lvl], width, label=lvl, color=colors[lvl], edgecolor="black", linewidth=0.6)
        ax.bar_label(rects, padding=3, fontsize=9)
        multiplier += 1
        
    ax.set_ylabel("Number of Positive Target Assignments")
    ax.set_title("Feature Level Positive Assignment Distribution (Starvation Effect)")
    ax.set_xticks(x + width * 1.5, stages)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 135)
    
    plt.tight_layout()
    plot_path = FORENSICS_DIR / "level_assignment_starvation.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_training_loss_and_map():
    """Compare training loss and validation mAP curves between Stage 0 and Stage 1."""
    s0_csv = ROOT / "results" / "experiments" / "visiontollnet_stage0" / "progress.csv"
    s1_csv = ROOT / "results" / "experiments" / "visiontollnet_stage1_p2" / "progress.csv"
    
    def read_csv(path):
        epochs, train_loss, cls_loss, map50_95 = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                epochs.append(int(r["epoch"]))
                train_loss.append(float(r["train_loss"]))
                cls_loss.append(float(r["cls_loss"]))
                map50_95.append(float(r["map50_95"]))
        return epochs, train_loss, cls_loss, map50_95

    s0_e, s0_tl, s0_cl, s0_map = read_csv(s0_csv)
    s1_e, s1_tl, s1_cl, s1_map = read_csv(s1_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train Loss plot (clip extreme epoch 2 spike for visibility, annotate it)
    ax1.plot(s0_e, s0_tl, "o-", color="#2b5c8f", label="Stage 0 Total Loss", linewidth=1.8, markersize=4)
    ax1.plot(s0_e, s0_cl, "--", color="#5c88b8", label="Stage 0 Cls Loss", linewidth=1.4)
    
    # Cap Stage 1 loss at 15 for display with annotation of the 114.3 spike
    s1_tl_clipped = [min(val, 15.0) for val in s1_tl]
    s1_cl_clipped = [min(val, 15.0) for val in s1_cl]
    
    ax1.plot(s1_e, s1_tl_clipped, "s-", color="#d55e00", label="Stage 1 Total Loss (P2)", linewidth=1.8, markersize=4)
    ax1.plot(s1_e, s1_cl_clipped, "--", color="#e69f00", label="Stage 1 Cls Loss (P2)", linewidth=1.4)
    ax1.annotate("Epoch 2 Spike:\nLoss=114.3\n(136k negative points)", xy=(2, 14.5), xytext=(3.5, 12),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 fontsize=8.5, backgroundcolor="#ffffff")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Dynamics (Stage 0 vs. Stage 1)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 16)
    
    # mAP50-95 plot
    ax2.plot(s0_e, s0_map, "o-", color="#2b5c8f", label="Stage 0 (mAP50-95: 0.3734)", linewidth=2.0, markersize=5)
    ax2.plot(s1_e, s1_map, "s-", color="#d55e00", label="Stage 1 + P2 (mAP50-95: 0.0144)", linewidth=2.0, markersize=5)
    ax2.annotate("Stage 1 Peak (Epoch 2):\nmAP=0.0144\nCollapse thereafter to 0.0000", xy=(2, 0.0144), xytext=(4, 0.12),
                 arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=5),
                 fontsize=8.5, backgroundcolor="#ffffff")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation mAP@0.50:0.95")
    ax2.set_title("Validation Metric Progression (mAP@0.50:0.95)")
    ax2.legend(loc="center right", frameon=True)
    ax2.set_ylim(-0.01, 0.42)
    
    plt.tight_layout()
    plot_path = FORENSICS_DIR / "training_dynamics_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_gradient_norms():
    """Plot gradient norms showing gradient starvation of P3, P4, P5."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    modules = ["Backbone", "Fusion P2", "Fusion P3", "Cls Head", "Reg Head"]
    s0_norms = [1.545, 0.000, 0.157, 1.053, 1.836]
    s1_best_norms = [1.971, 0.149, 0.113, 1.188, 2.421]
    # For s1_last, cls head was 102.99; we plot log scale or bar with break
    s1_last_norms = [4.940, 0.619, 0.000, 102.995, 23.642]
    
    x = np.arange(len(modules))
    width = 0.25
    
    r1 = ax.bar(x - width, s0_norms, width, label="Stage 0 (Baseline)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x, s1_best_norms, width, label="Stage 1 (Epoch 2 - Best)", color="#389e82", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + width, s1_last_norms, width, label="Stage 1 (Epoch 20 - Last)", color="#d55e00", edgecolor="black", linewidth=0.5)
    
    ax.set_yscale("log")
    ax.set_ylabel("Gradient L2 Norm (log scale)")
    ax.set_title("Submodule Gradient Norms (Demonstrating Head Destabilization & Level Death)")
    ax.set_xticks(x, modules)
    ax.legend(loc="upper left", frameon=True)
    
    # Annotations
    ax.annotate("Fusion P3 Grad = 0.0\n(Dead Level)", xy=(2 + width, 0.001), xytext=(1.8, 0.02),
                 arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=4),
                 fontsize=8, backgroundcolor="#ffffff")
    
    plt.tight_layout()
    plot_path = FORENSICS_DIR / "gradient_norm_starvation.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    plot_assignment_starvation()
    plot_training_loss_and_map()
    plot_gradient_norms()


if __name__ == "__main__":
    main()
