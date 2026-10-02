import os
import sys
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

FGVD_ROOT = "D:/VisionToll/data/raw/FGVD/IDD_FGVD"
RESULTS_DIR = "D:/VisionToll/results/dataset_audit"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80, flush=True)
print("FAST FGVD RAW DATASET INSPECTOR", flush=True)
print("=" * 80, flush=True)
print(f"Dataset root: {FGVD_ROOT}\n", flush=True)

splits = ["train", "val", "test"]
split_stats = {}
all_labels = Counter()
level1_types = Counter()
level2_makes = Counter()
level3_models = Counter()
split_level1 = defaultdict(Counter)

total_xmls = 0
total_images = 0
total_boxes = 0
image_resolutions = Counter()
invalid_boxes = 0
out_of_bounds = 0

for split in splits:
    split_dir = os.path.join(FGVD_ROOT, split)
    anno_dir = os.path.join(split_dir, "annos")
    img_dir = os.path.join(split_dir, "images")

    if not os.path.exists(anno_dir) or not os.path.exists(img_dir):
        print(f"Directory missing for split: {split}", flush=True)
        continue

    xml_files = [f for f in os.listdir(anno_dir) if f.lower().endswith(".xml")]
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_set = set(img_files)

    split_boxes = 0
    missing_for_split = 0

    print(f"Processing split '{split}': {len(xml_files)} XMLs, {len(img_files)} images...", flush=True)

    for xml_name in xml_files:
        xml_path = os.path.join(anno_dir, xml_name)
        total_xmls += 1

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_el = root.find("filename")
            filename = filename_el.text.strip() if filename_el is not None and filename_el.text else os.path.splitext(xml_name)[0] + ".jpg"

            if filename not in img_set:
                stem_jpg = os.path.splitext(xml_name)[0] + ".jpg"
                if stem_jpg not in img_set:
                    missing_for_split += 1

            size_el = root.find("size")
            img_w, img_h = None, None
            if size_el is not None:
                w_el = size_el.find("width")
                h_el = size_el.find("height")
                if w_el is not None and h_el is not None and w_el.text and h_el.text:
                    img_w = int(w_el.text.strip())
                    img_h = int(h_el.text.strip())
                    image_resolutions[(img_w, img_h)] += 1

            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is None or not name_el.text:
                    continue
                label = name_el.text.strip()
                all_labels[label] += 1
                total_boxes += 1
                split_boxes += 1

                parts = label.split("_")
                l1 = parts[0]
                l2 = parts[1] if len(parts) > 1 else "Unknown"
                l3 = "_".join(parts[2:]) if len(parts) > 2 else "Unknown"

                level1_types[l1] += 1
                level2_makes[l2] += 1
                if l3 != "Unknown":
                    level3_models[l3] += 1
                split_level1[split][l1] += 1

                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    try:
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        if xmax <= xmin or ymax <= ymin:
                            invalid_boxes += 1
                        if img_w is not None and img_h is not None:
                            if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                                out_of_bounds += 1
                    except Exception:
                        invalid_boxes += 1

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}", flush=True)

    split_stats[split] = {
        "xmls": len(xml_files),
        "images": len(img_files),
        "boxes": split_boxes,
        "missing_images": missing_for_split
    }
    total_images += len(img_files)

print("\n" + "=" * 80, flush=True)
print("SPLIT SUMMARY", flush=True)
print("=" * 80, flush=True)
for sp in splits:
    info = split_stats.get(sp, {})
    print(f"  {sp.upper():5s} -> Images: {info.get('images', 0):5d} | XMLs: {info.get('xmls', 0):5d} | Boxes: {info.get('boxes', 0):6d} | Missing Imgs: {info.get('missing_images', 0)}", flush=True)
print(f"TOTAL : Images: {total_images} | XMLs: {total_xmls} | Boxes: {total_boxes}\n", flush=True)

print("=" * 80, flush=True)
print(f"LEVEL-1 (BROAD CATEGORY) DISTRIBUTION ({len(level1_types)} unique types)", flush=True)
print("=" * 80, flush=True)
for l1, cnt in level1_types.most_common():
    pct = (cnt / max(total_boxes, 1)) * 100
    print(f"  {l1:25s}: {cnt:6d} boxes ({pct:5.2f}%)", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"TOP 50 FULL FINE-GRAINED LABELS ({len(all_labels)} total unique labels)", flush=True)
print("=" * 80, flush=True)
for lbl, cnt in all_labels.most_common(50):
    print(f"  {lbl:45s}: {cnt:5d}", flush=True)

print("\n" + "=" * 80, flush=True)
print("DATA INTEGRITY AUDIT", flush=True)
print("=" * 80, flush=True)
print(f"Inverted Boxes (w<=0 or h<=0): {invalid_boxes}", flush=True)
print(f"Out of Image Bounds Boxes    : {out_of_bounds}", flush=True)
print(f"Image Resolutions            : {dict(image_resolutions)}", flush=True)

# Save JSON report
audit_report = {
    "total_images": total_images,
    "total_xmls": total_xmls,
    "total_boxes": total_boxes,
    "splits": split_stats,
    "level1_types": dict(level1_types),
    "level2_makes": dict(level2_makes.most_common(50)),
    "top_fine_grained_labels": dict(all_labels.most_common(100)),
    "all_fine_grained_labels": dict(all_labels),
    "integrity": {
        "invalid_boxes": invalid_boxes,
        "out_of_bounds": out_of_bounds,
        "image_resolutions": {f"{k[0]}x{k[1]}": v for k, v in image_resolutions.items()}
    }
}

out_file = os.path.join(RESULTS_DIR, "fgvd_raw_inspection.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(audit_report, f, indent=2)
print(f"\nInspection JSON saved to: {out_file}", flush=True)
