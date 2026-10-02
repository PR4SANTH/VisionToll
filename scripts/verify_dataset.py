"""
VisionToll Visual Verification & Contact Sheet Generator
Samples images from each of the 5 target classes, overlays ground-truth bounding boxes,
and generates annotated inspection images and a composite contact sheet.
"""

import os
import sys
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
OUT_DIR = ROOT / "results" / "dataset_audit" / "visual_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

CLASS_COLORS = {
    0: (255, 120, 0),    # Blue/Orange in BGR: Bus
    1: (0, 200, 0),      # Green: Car
    2: (0, 0, 255),      # Red: Motorcycle
    3: (255, 0, 255),    # Magenta: Auto Rickshaw
    4: (0, 215, 255)     # Yellow/Gold: Truck
}

def verify_dataset(data_root=DATA_DIR, samples_per_class=4, seed=42):
    random.seed(seed)
    print("=" * 80)
    print("VISIONTOLL: VISUAL GROUND-TRUTH VERIFICATION")
    print("=" * 80)
    print(f"Data directory: {data_root}")
    print(f"Output preview: {OUT_DIR}\n")

    if not data_root.exists():
        print(f"Error: {data_root} does not exist!")
        return

    # Index images by class
    class_to_images = defaultdict(list)
    lbl_dir = data_root / "labels" / "train"
    img_dir = data_root / "images" / "train"

    if not lbl_dir.exists():
        print(f"Labels directory {lbl_dir} not found!")
        return

    # Fast direct stem mapping
    for lbl_file in lbl_dir.glob("*.txt"):
        img_p = img_dir / (lbl_file.stem + ".jpg")
        if not img_p.exists():
            continue

        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    if cid in CLASS_NAMES:
                        class_to_images[cid].append((img_p, lbl_file))
                except ValueError:
                    pass

    # Sample and render previews per class
    sampled_tiles = []
    
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        candidates = class_to_images[cid]
        print(f"Class {cid} ({cname}): {len(candidates)} candidate images")
        if not candidates:
            continue

        # Use deterministic random sample
        selected = random.sample(candidates, min(samples_per_class, len(candidates)))

        for idx, (img_path, lbl_path) in enumerate(selected):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    box_cid = int(parts[0])
                    xc, yc, bw, bh = [float(v) for v in parts[1:]]

                    xmin = int((xc - bw / 2.0) * w)
                    ymin = int((yc - bh / 2.0) * h)
                    xmax = int((xc + bw / 2.0) * w)
                    ymax = int((yc + bh / 2.0) * h)

                    color = CLASS_COLORS.get(box_cid, (255, 255, 255))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                    label_str = CLASS_NAMES.get(box_cid, str(box_cid))
                    cv2.putText(img, label_str, (xmin, max(ymin - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add header on image
            header = f"{cname} | {img_path.name}"
            cv2.rectangle(img, (0, 0), (w, 32), (0, 0, 0), -1)
            cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            out_img_path = OUT_DIR / f"sample_c{cid}_{cname.replace(' ', '_')}_{idx+1}.jpg"
            cv2.imwrite(str(out_img_path), img)

            # Resize for contact sheet tile
            tile = cv2.resize(img, (320, 240))
            sampled_tiles.append(tile)

    # Sample difficult scenarios (crowded scenes and small objects)
    print("\nSampling difficult scenarios (crowded scenes & small objects)...")
    crowded_candidates = []
    small_obj_candidates = []

    for lbl_file in lbl_dir.glob("*.txt"):
        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) >= 8:
            crowded_candidates.append(lbl_file)
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                bw, bh = float(parts[3]), float(parts[4])
                if (bw * bh) < (32 * 32 / (1920 * 1080)):
                    small_obj_candidates.append(lbl_file)
                    break

    # Render crowded samples
    if crowded_candidates:
        for idx, lbl_file in enumerate(random.sample(crowded_candidates, min(2, len(crowded_candidates))), 1):
            img_path = img_dir / (lbl_file.stem + ".jpg")
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    box_cid = int(parts[0])
                    xc, yc, bw, bh = [float(v) for v in parts[1:]]
                    xmin = int((xc - bw / 2.0) * w)
                    ymin = int((yc - bh / 2.0) * h)
                    xmax = int((xc + bw / 2.0) * w)
                    ymax = int((yc + bh / 2.0) * h)
                    color = CLASS_COLORS.get(box_cid, (255, 255, 255))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    label_str = CLASS_NAMES.get(box_cid, str(box_cid))
                    cv2.putText(img, label_str, (xmin, max(ymin - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                header = f"CROWDED TRAFFIC ({len(lines)} objs) | {img_path.name}"
                cv2.rectangle(img, (0, 0), (w, 32), (0, 0, 0), -1)
                cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
                cv2.imwrite(str(OUT_DIR / f"difficult_crowded_{idx}.jpg"), img)
                sampled_tiles.append(cv2.resize(img, (320, 240)))

    # Render small object samples
    if small_obj_candidates:
        for idx, lbl_file in enumerate(random.sample(small_obj_candidates, min(2, len(small_obj_candidates))), 1):
            img_path = img_dir / (lbl_file.stem + ".jpg")
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    box_cid = int(parts[0])
                    xc, yc, bw, bh = [float(v) for v in parts[1:]]
                    xmin = int((xc - bw / 2.0) * w)
                    ymin = int((yc - bh / 2.0) * h)
                    xmax = int((xc + bw / 2.0) * w)
                    ymax = int((yc + bh / 2.0) * h)
                    color = CLASS_COLORS.get(box_cid, (255, 255, 255))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    label_str = CLASS_NAMES.get(box_cid, str(box_cid))
                    cv2.putText(img, label_str, (xmin, max(ymin - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                header = f"SMALL DISTANT VEHICLE | {img_path.name}"
                cv2.rectangle(img, (0, 0), (w, 32), (0, 0, 0), -1)
                cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
                cv2.imwrite(str(OUT_DIR / f"difficult_small_{idx}.jpg"), img)
                sampled_tiles.append(cv2.resize(img, (320, 240)))

    # Build contact sheet (grid)
    if sampled_tiles:
        num_cols = 4
        num_rows = (len(sampled_tiles) + num_cols - 1) // num_cols

        # Pad tiles to fill grid
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        while len(sampled_tiles) < num_rows * num_cols:
            sampled_tiles.append(blank)

        row_images = []
        for r in range(num_rows):
            row_tiles = sampled_tiles[r * num_cols:(r + 1) * num_cols]
            row_images.append(np.hstack(row_tiles))
        contact_sheet = np.vstack(row_images)

        contact_path = OUT_DIR.parent / "contact_sheet.jpg"
        cv2.imwrite(str(contact_path), contact_sheet)
        print(f"\nContact sheet saved to: {contact_path}")

    print("=" * 80)
    print("VISUAL VERIFICATION COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    verify_dataset()
