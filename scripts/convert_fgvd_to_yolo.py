"""
VisionToll FGVD to YOLO Conversion Pipeline
Converts Pascal-VOC XML annotations to normalized YOLO format (class x_center y_center width height)
Strictly adheres to the 5 target classes:
  0 = Bus
  1 = Car
  2 = Motorcycle
  3 = Auto Rickshaw
  4 = Truck

Explicitly excludes:
  - Scooter (must NOT be mapped to Motorcycle)
  - Mini-bus (must NOT be mapped to Bus)
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
import cv2

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
RAW_DIR = ROOT / "data" / "raw" / "FGVD" / "IDD_FGVD"
PROCESSED_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"

TARGET_CLASSES = {
    "Bus": 0,
    "Car": 1,
    "Motorcycle": 2,
    "Auto Rickshaw": 3,
    "Truck": 4,
}

CLASS_NAMES = {v: k for k, v in TARGET_CLASSES.items()}

# Verified FGVD prefix to target class mapping
FGVD_PREFIX_TO_CLASS = {
    "bus": "Bus",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "autorickshaw": "Auto Rickshaw",
    "truck": "Truck",
    "scooter": None,     # Explicitly excluded
    "mini-bus": None,    # Explicitly excluded
}

def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Converts Pascal VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, width, height)
    normalized to [0, 1].
    Safely clips coordinates to image bounds to handle 1-pixel out-of-bounds coordinates.
    """
    xmin = max(0.0, min(float(xmin), float(img_w)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    ymax = max(0.0, min(float(ymax), float(img_h)))

    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w <= 1.0 or box_h <= 1.0:
        return None  # degenerate box

    x_center = (xmin + xmax) / (2.0 * img_w)
    y_center = (ymin + ymax) / (2.0 * img_h)
    norm_w = box_w / img_w
    norm_h = box_h / img_h

    # Ensure strictly within [0.0, 1.0]
    x_center = max(0.0, min(x_center, 1.0))
    y_center = max(0.0, min(y_center, 1.0))
    norm_w = max(0.0, min(norm_w, 1.0))
    norm_h = max(0.0, min(norm_h, 1.0))

    return x_center, y_center, norm_w, norm_h

def convert_dataset(raw_root=RAW_DIR, out_root=PROCESSED_DIR, include_empty=True):
    """
    Converts Pascal VOC annotations into YOLO format using the verified 5-class mapping.
    Preserves train, val, and test splits and 1-to-1 image/label correspondence.
    """
    print("=" * 80)
    print("VISIONTOLL: CONVERTING FGVD (PASCAL VOC) TO YOLOV8 FORMAT")
    print("=" * 80)
    print(f"Source RAW FGVD : {raw_root}")
    print(f"Target Processed: {out_root}")
    print(f"Target Classes  : {TARGET_CLASSES}\n")

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_root}")

    # Prepare directories
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "total_xmls": 0,
        "copied_images": 0,
        "total_source_boxes": 0,
        "mapped_target_boxes": 0,
        "excluded_boxes": 0,
        "invalid_boxes": 0,
        "corrupt_images": 0,
        "class_counts": Counter(),
        "split_counts": defaultdict(lambda: {"images": 0, "boxes": 0, "empty_images": 0, "class_counts": Counter()}),
    }

    # Discover and sort all XML files for deterministic processing
    all_xmls = sorted(list(raw_root.rglob("*.xml")))
    stats["total_xmls"] = len(all_xmls)
    print(f"Found {len(all_xmls)} XML annotations to process.")

    for idx, xml_path in enumerate(all_xmls, 1):
        if idx % 500 == 0 or idx == len(all_xmls):
            print(f"  Processed {idx}/{len(all_xmls)} XMLs...", end="\r", flush=True)

        # Determine split from path hierarchy
        parts = [p.lower() for p in xml_path.parts]
        split = "train"
        for s in ["train", "val", "test"]:
            if s in parts:
                split = s
                break

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_elem = root.find("filename")
            filename = filename_elem.text.strip() if filename_elem is not None and filename_elem.text else xml_path.stem + ".jpg"

            # Locate source image
            img_path = xml_path.parent.parent / "images" / filename
            if not img_path.exists():
                img_path = xml_path.parent.parent / "images" / (xml_path.stem + ".jpg")

            if not img_path.exists():
                # Broader search fallback
                cands = list(xml_path.parent.parent.rglob(filename))
                if cands:
                    img_path = cands[0]

            if not img_path.exists():
                print(f"Warning: Image not found for {xml_path}")
                continue

            # Read image dimensions from XML
            size_elem = root.find("size")
            img_w, img_h = None, None
            if size_elem is not None:
                w_el = size_elem.find("width")
                h_el = size_elem.find("height")
                if w_el is not None and h_el is not None and w_el.text and h_el.text:
                    img_w = int(w_el.text.strip())
                    img_h = int(h_el.text.strip())

            if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["corrupt_images"] += 1
                    continue
                img_h, img_w = img.shape[:2]

            yolo_lines = []

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None or not name_elem.text:
                    continue
                raw_label = name_elem.text.strip()
                stats["total_source_boxes"] += 1

                # Parse prefix before first underscore
                prefix = raw_label.split("_")[0].lower()
                target_class_name = FGVD_PREFIX_TO_CLASS.get(prefix, None)

                # Strict validation: exclude scooters, mini-buses, and unknown labels
                if target_class_name is None or target_class_name not in TARGET_CLASSES:
                    stats["excluded_boxes"] += 1
                    continue

                class_id = TARGET_CLASSES[target_class_name]

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                except Exception:
                    stats["invalid_boxes"] += 1
                    continue

                yolo_box = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
                if yolo_box is None:
                    stats["invalid_boxes"] += 1
                    continue

                xc, yc, nw, nh = yolo_box
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                stats["mapped_target_boxes"] += 1
                stats["class_counts"][target_class_name] += 1
                stats["split_counts"][split]["boxes"] += 1
                stats["split_counts"][split]["class_counts"][target_class_name] += 1

            if len(yolo_lines) == 0 and not include_empty:
                continue

            # Write converted image and label
            dest_img_path = out_root / "images" / split / img_path.name
            dest_lbl_path = out_root / "labels" / split / (img_path.stem + ".txt")

            # Only copy if file doesn't already exist or has different size (idempotency)
            if not dest_img_path.exists() or dest_img_path.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dest_img_path)

            lbl_content = "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
            dest_lbl_path.write_text(lbl_content, encoding="utf-8")

            stats["copied_images"] += 1
            stats["split_counts"][split]["images"] += 1
            if len(yolo_lines) == 0:
                stats["split_counts"][split]["empty_images"] += 1

        except Exception as e:
            print(f"\nError converting {xml_path}: {e}")

    # Generate data.yaml with relative path portable for local environments
    data_yaml_content = """# VisionToll YOLOv8 Dataset Configuration
# Academic Project: Deep Learning Course (23CSE473)
path: data/processed/vision_toll_yolo
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: Bus
  1: Car
  2: Motorcycle
  3: Auto Rickshaw
  4: Truck
"""
    (out_root / "data.yaml").write_text(data_yaml_content, encoding="utf-8")

    print("\n\n" + "=" * 80)
    print("CONVERSION COMPLETE SUMMARY")
    print("=" * 80)
    print(f"Total Source XMLs      : {stats['total_xmls']}")
    print(f"Total Processed Images : {stats['copied_images']}")
    print(f"Total Source Boxes     : {stats['total_source_boxes']}")
    print(f"Mapped Target Boxes    : {stats['mapped_target_boxes']}")
    print(f"Excluded Boxes         : {stats['excluded_boxes']}")
    print(f"Invalid / Degenerate   : {stats['invalid_boxes']}")
    print(f"Corrupt Images         : {stats['corrupt_images']}")

    print("\nPer-Split Breakdown:")
    for sp in ["train", "val", "test"]:
        sp_data = stats["split_counts"][sp]
        print(f"  {sp.upper():5s} -> Images: {sp_data['images']:5d} (Empty: {sp_data['empty_images']:3d}) | Boxes: {sp_data['boxes']:5d}")
        for cid in range(5):
            cls_name = CLASS_NAMES[cid]
            ccount = sp_data["class_counts"][cls_name]
            print(f"         Class {cid} ({cls_name:15s}): {ccount:5d}")

    print("\nFinal 5-Class Target Ground Truth Totals:")
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        cnt = stats["class_counts"][cname]
        pct = (cnt / max(stats["mapped_target_boxes"], 1)) * 100
        print(f"  Class {cid} ({cname:15s}): {cnt:6d} boxes ({pct:5.2f}%)")

    print(f"\nGenerated data.yaml at: {out_root / 'data.yaml'}")
    return stats

if __name__ == "__main__":
    convert_dataset()
