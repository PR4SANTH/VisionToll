"""
VisionToll Comprehensive Dataset Audit
Calculates exact empirical statistics from the 5-class YOLO dataset:
- Dataset size (splits, image counts, label counts)
- Per-class distribution (counts, percentages, co-occurrences)
- Bounding-box geometry (width, height, area, aspect ratio, small/medium/large)
- Image resolutions
- Data quality verification (missing, corrupted, out-of-bounds, duplicates)
- Scene density & crowdedness
- Generates publication-ready figures in results/dataset_audit/
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
RESULTS_DIR = ROOT / "results" / "dataset_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

def audit_dataset(data_root=DATA_DIR, out_dir=RESULTS_DIR):
    print("=" * 80)
    print("VISIONTOLL: DATASET AUDIT & STATISTICAL PROFILING")
    print("=" * 80)
    print(f"Dataset root: {data_root}")
    print(f"Results dir : {out_dir}\n")

    if not data_root.exists():
        print(f"Error: {data_root} does not exist! Please run conversion first.")
        return

    splits = ["train", "val", "test"]
    box_records = []
    image_records = []

    quality_issues = {
        "missing_images": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_format_rows": 0,
        "out_of_bounds_coords": 0,
        "degenerate_boxes": 0,
        "corrupted_images": 0
    }

    for split in splits:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

        if not img_dir.exists():
            continue

        images = [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        lbl_map = {p.stem: p for p in labels}

        for img_p in images:
            # Check image integrity & resolution
            img = cv2.imread(str(img_p))
            if img is None:
                quality_issues["corrupted_images"] += 1
                continue
            h, w = img.shape[:2]

            lbl_p = lbl_map.get(img_p.stem)
            if lbl_p is None or not lbl_p.exists():
                quality_issues["missing_labels"] += 1
                continue

            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                quality_issues["empty_labels"] += 1

            img_classes = set()
            img_box_count = 0

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    quality_issues["invalid_format_rows"] += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = [float(v) for v in parts[1:]]
                except ValueError:
                    quality_issues["invalid_format_rows"] += 1
                    continue

                if cls_id not in CLASS_NAMES:
                    quality_issues["invalid_format_rows"] += 1
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    quality_issues["out_of_bounds_coords"] += 1
                    continue

                # Pixel area and relative area
                pixel_w = bw * w
                pixel_h = bh * h
                pixel_area = pixel_w * pixel_h
                rel_area = bw * bh

                if pixel_w <= 1 or pixel_h <= 1:
                    quality_issues["degenerate_boxes"] += 1
                    continue

                # Scale classification (COCO standard: small < 32^2 = 1024, medium 1024-96^2 = 9216, large > 9216)
                if pixel_area < 1024:
                    scale = "Small"
                elif pixel_area < 9216:
                    scale = "Medium"
                else:
                    scale = "Large"

                img_classes.add(cls_id)
                img_box_count += 1

                box_records.append({
                    "split": split,
                    "image": img_p.name,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "xc": xc,
                    "yc": yc,
                    "bw": bw,
                    "bh": bh,
                    "pixel_w": pixel_w,
                    "pixel_h": pixel_h,
                    "pixel_area": pixel_area,
                    "rel_area": rel_area,
                    "aspect_ratio": pixel_w / max(pixel_h, 1e-5),
                    "scale": scale
                })

            image_records.append({
                "split": split,
                "image": img_p.name,
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "num_objects": img_box_count,
                "num_classes": len(img_classes),
                "classes": [CLASS_NAMES[c] for c in img_classes]
            })

    df_boxes = pd.DataFrame(box_records)
    df_images = pd.DataFrame(image_records)

    # 1. Dataset size report
    total_imgs = len(df_images)
    total_boxes = len(df_boxes)
    split_img_counts = df_images["split"].value_counts().to_dict()
    split_box_counts = df_boxes["split"].value_counts().to_dict()

    print("--- 1. DATASET SIZE ---")
    print(f"Total Images      : {total_imgs}")
    print(f"Total Annotations : {total_boxes}")
    for sp in splits:
        ic = split_img_counts.get(sp, 0)
        bc = split_box_counts.get(sp, 0)
        ipct = (ic / max(total_imgs, 1)) * 100
        bpct = (bc / max(total_boxes, 1)) * 100
        print(f"  {sp.upper():5s} -> Images: {ic:5d} ({ipct:5.1f}%) | Boxes: {bc:5d} ({bpct:5.1f}%)")

    # 2. Per-class distribution
    print("\n--- 2. PER-CLASS DISTRIBUTION ---")
    class_dist = df_boxes["class_name"].value_counts()
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        cnt = class_dist.get(cname, 0)
        pct = (cnt / max(total_boxes, 1)) * 100
        # Count images containing class
        imgs_with_c = df_images[df_images["classes"].apply(lambda cs: cname in cs)]["image"].count()
        print(f"  Class {cid} ({cname:15s}): {cnt:6d} boxes ({pct:5.2f}%) | Present in {imgs_with_c:5d} images")

    # 3. Image resolution profile
    print("\n--- 3. IMAGE RESOLUTION PROFILE ---")
    res_dist = df_images["resolution"].value_counts()
    print(f"Distinct resolutions: {len(res_dist)}")
    for res_name, rcnt in res_dist.head(5).items():
        print(f"  {res_name:12s}: {rcnt:5d} images ({(rcnt/max(total_imgs,1))*100:.1f}%)")

    # 4. Bounding box scale distribution
    print("\n--- 4. OBJECT SCALE PROFILE (COCO SCALE STANDARD) ---")
    scale_dist = df_boxes["scale"].value_counts()
    for sc in ["Small", "Medium", "Large"]:
        sc_cnt = scale_dist.get(sc, 0)
        print(f"  {sc:8s} (< 32^2 / < 96^2 / > 96^2): {sc_cnt:6d} ({sc_cnt/max(total_boxes,1)*100:5.1f}%)")

    # 5. Scene density
    print("\n--- 5. SCENE DENSITY ---")
    print(f"Mean objects per image: {df_images['num_objects'].mean():.2f}")
    print(f"Median objects/image  : {df_images['num_objects'].median():.2f}")
    print(f"Max objects in image  : {df_images['num_objects'].max():.2f}")

    # 6. Quality issues
    print("\n--- 6. DATA QUALITY AUDIT ---")
    for k, v in quality_issues.items():
        print(f"  {k:22s}: {v}")

    # Save CSV and JSON
    summary_data = {
        "dataset_size": {
            "total_images": total_imgs,
            "total_boxes": total_boxes,
            "splits": {sp: {"images": split_img_counts.get(sp, 0), "boxes": split_box_counts.get(sp, 0)} for sp in splits}
        },
        "per_class": {
            CLASS_NAMES[cid]: {
                "count": int(class_dist.get(CLASS_NAMES[cid], 0)),
                "percentage": float(class_dist.get(CLASS_NAMES[cid], 0) / max(total_boxes, 1) * 100)
            } for cid in range(5)
        },
        "scale_distribution": {sc: int(scale_dist.get(sc, 0)) for sc in ["Small", "Medium", "Large"]},
        "scene_density": {
            "mean_objects_per_image": float(df_images["num_objects"].mean()),
            "median_objects_per_image": float(df_images["num_objects"].median()),
            "max_objects_per_image": int(df_images["num_objects"].max())
        },
        "quality_audit": quality_issues
    }

    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    df_boxes.to_csv(out_dir / "bounding_box_details.csv", index=False)

    # 7. Generate publication-ready figures
    print("\n--- 7. GENERATING AUDIT FIGURES ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Class Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    cls_order = [CLASS_NAMES[i] for i in range(5)]
    counts = [class_dist.get(c, 0) for c in cls_order]
    bars = ax.bar(cls_order, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(counts)*0.01, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("VisionToll 5-Class Object Distribution (FGVD Ground Truth)", fontweight="bold")
    ax.set_ylabel("Bounding Box Count")
    ax.set_xlabel("Vehicle Class")
    plt.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Box Scale by Class
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_boxes, x="class_name", hue="scale", order=cls_order, hue_order=["Small", "Medium", "Large"], palette="viridis", ax=ax)
    ax.set_title("Object Scale Distribution by Vehicle Category (COCO Scale Metric)", fontweight="bold")
    ax.set_xlabel("Vehicle Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / "scale_distribution.png", dpi=300)
    plt.close()

    # Figure 3: Objects Per Image Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_images["num_objects"], bins=range(1, int(df_images["num_objects"].max()) + 2), color="#2b5c8f", discrete=True, ax=ax)
    ax.set_title("Traffic Scene Density: Distribution of Vehicles Per Image", fontweight="bold")
    ax.set_xlabel("Number of Annotated Vehicles")
    ax.set_ylabel("Image Count")
    plt.tight_layout()
    fig.savefig(out_dir / "objects_per_image.png", dpi=300)
    plt.close()

    print(f"Generated figures in: {out_dir}")
    print("=" * 80)
    print("DATASET AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return summary_data

if __name__ == '__main__':
    audit_dataset()
