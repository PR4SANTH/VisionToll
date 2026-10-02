# VisionToll Dataset Specification & Verified Class Mapping

## 1. Dataset Overview: IIIT-H FGVD

- **Official Release**: Fine-Grained Vehicle Detection (FGVD) Dataset for Unconstrained Roads (ICVGIP 2022)
- **Source Repository**: Zenodo Record [7488960](https://zenodo.org/records/7488960)
- **Archive**: `IDD_FGVD.tar.gz` (2,756,233,295 bytes, MD5: `e0f5d69c0e766b1135ec437ad950c911`)
- **Annotation Format**: Pascal VOC XML (converted to normalized YOLOv8 format)
- **Raw Hierarchy**: 3-level encoding in `<name>` tags: `VehicleType_Manufacturer_Model`
- **Total Raw Images**: 5,502 scene images (Train: 3,535 | Val: 884 | Test: 1,083)
- **Total Raw XMLs**: 5,502 XML files (100% paired with images, 0 missing)
- **Total Raw Bounding Boxes**: 24,450 annotations

--- 

## 2. High-Level Category Mapping Summary

| Level-1 Source Prefix | Bounding Boxes | VisionToll Class ID | VisionToll Class Name | Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bus` | 1,215 | `0` | **Bus** | **INCLUDED** | Direct semantic match for passenger transit buses. |
| `car` | 7,951 | `1` | **Car** | **INCLUDED** | All passenger sedans, hatchbacks, SUVs, and MUVs. |
| `motorcycle` | 5,293 | `2` | **Motorcycle** | **INCLUDED** | Standard two-wheeled motorcycles (step-over frame). |
| `autorickshaw` | 3,777 | `3` | **Auto Rickshaw** | **INCLUDED** | Normalized from `autorickshaw` to `Auto Rickshaw`. |
| `truck` | 1,552 | `4` | **Truck** | **INCLUDED** | Heavy commercial freight vehicles, tippers, haulers. |
| `scooter` | 4,347 | N/A | *Excluded* | **EXCLUDED** | Non-negotiable rule: Scooters are morphologically distinct step-through two-wheelers and must not be relabeled as Motorcycle. |
| `mini-bus` | 315 | N/A | *Excluded* | **EXCLUDED** | Non-negotiable rule: Excluded to prevent semantic dilution of heavy transit Bus class. |

### Summary Statistics for Converted 5-Class Target Ground Truth:
- **Total Converted Images**: **5,502** images (Train: 3,535 | Val: 884 | Test: 1,083)
- **Images with ≥1 Target Object**: **5,438** images (98.84%)
- **Background / Empty-Label Images**: **64** images (1.16%: 34 Train | 17 Val | 13 Test)
- **Retained Target Boxes**: **19,788** (80.93% of raw annotations)
- **Excluded Non-Target Boxes**: **4,662** (19.07% of raw annotations: 4,347 Scooters + 315 Mini-buses)
- **Degenerate / Invalid Boxes**: **0** (100% valid geometry)

---

## 3. Converted Split & Per-Class Distribution

| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes | Class % | Image Presence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Bus** | 797 | 199 | 219 | **1,215** | 6.14% | 970 images |
| **1** | **Car** | 5,114 | 1,278 | 1,559 | **7,951** | 40.18% | 3,843 images |
| **2** | **Motorcycle** | 3,428 | 780 | 1,085 | **5,293** | 26.75% | 3,059 images |
| **3** | **Auto Rickshaw** | 2,413 | 610 | 754 | **3,777** | 19.09% | 2,314 images |
| **4** | **Truck** | 1,010 | 233 | 309 | **1,552** | 7.84% | 1,311 images |
| **Total** | — | **12,762** | **3,100** | **3,926** | **19,788** | **100.00%** | — |

---

## 4. Empirical Geometry & Scale Profile

Empirical statistics computed via `scripts/audit_dataset.py`:

- **Image Resolutions**:
  - `1920x1080`: 4,424 images (80.4%)
  - `1921x1080`: 665 images (12.1%)
  - `1280x720`: 413 images (7.5%)
- **Object Scale Distribution (COCO Standard)**:
  - **Small** ($< 32^2 = 1,024 \text{ px}^2$): 131 boxes (0.7%)
  - **Medium** ($1,024 \text{ to } 96^2 = 9,216 \text{ px}^2$): 2,144 boxes (10.8%)
  - **Large** ($> 96^2 = 9,216 \text{ px}^2$): 17,513 boxes (88.5%)
- **Traffic Scene Density**:
  - Mean objects per image: 3.60
  - Median objects per image: 3.00
  - Maximum objects in a single image: 17.00

---

## 5. Verified Label-to-Class Mapping Tables

### Class 0: Bus (1,215 Boxes, 1 Unique FGVD Label)

| FGVD Source Label | Bounding Box Count | VisionToll Target Class | Mapping Rule |
| :--- | :--- | :--- | :--- |
| `bus` | 1,215 | `Bus` (0) | Exact semantic match |

### Class 4: Truck (1,552 Boxes, 7 Unique FGVD Labels)

| FGVD Source Label | Bounding Box Count | VisionToll Target Class | Mapping Rule |
| :--- | :--- | :--- | :--- |
| `truck_Tata` | 552 | `Truck` (4) | Prefix `truck` match |
| `truck_Others` | 425 | `Truck` (4) | Prefix `truck` match |
| `truck_Eicher` | 249 | `Truck` (4) | Prefix `truck` match |
| `truck_Mahindra` | 156 | `Truck` (4) | Prefix `truck` match |
| `truck_AshokLeyland` | 142 | `Truck` (4) | Prefix `truck` match |
| `truck_BharatBenz` | 25 | `Truck` (4) | Prefix `truck` match |
| `truck_SML` | 3 | `Truck` (4) | Prefix `truck` match |

### Class 3: Auto Rickshaw (3,777 Boxes, 7 Unique FGVD Labels)

| FGVD Source Label | Bounding Box Count | VisionToll Target Class | Mapping Rule |
| :--- | :--- | :--- | :--- |
| `autorickshaw_Others` | 2,408 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_Bajaj` | 969 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_Piaggio` | 259 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_TVS` | 88 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_Atul` | 24 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_Covered` | 15 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |
| `autorickshaw_Mahindra` | 14 | `Auto Rickshaw` (3) | Prefix `autorickshaw` match |

### Class 2: Motorcycle (5,293 Boxes, 67 Unique FGVD Labels)

Top models: `motorcycle_Hero_Splendor` (756), `motorcycle_Bajaj_Pulsar150` (618), `motorcycle_Others` (584), `motorcycle_Honda_Shine` (448), `motorcycle_Hero_PassionPro` (319), `motorcycle_TVS_XL100` (296), `motorcycle_Hero_Glamour` (226), `motorcycle_Hero_Passion` (210), `motorcycle_RoyalEnfield_Classic350` (205), etc. All 67 fine-grained types map directly to `Motorcycle` (Class 2).

### Class 1: Car (7,951 Boxes, 110 Unique FGVD Labels)

Top models: `car_MarutiSuzuki_Dzire` (788), `car_TataMotors_Indica` (747), `car_Others` (553), `car_Toyota_Innova` (500), `car_Toyota_Etios` (484), `car_MarutiSuzuki_Swift` (454), `car_MarutiSuzuki_Alto800` (378), `car_MarutiSuzuki_Omni` (258), `car_Hyundai_I10` (195), `car_Ford_Figo` (186), `car_Hyundai_Santro` (180), `car_MarutiSuzuki_Ritz` (180), `car_MarutiSuzuki_WagonR` (178), etc. All 110 fine-grained car types map directly to `Car` (Class 1).

### Excluded Classes (4,662 Boxes)

| FGVD Source Category | Bounding Box Count | Status | Formal Academic Exclusion Reason |
| :--- | :--- | :--- | :--- |
| `scooter` (22 makes/models) | 4,347 | **EXCLUDED** | Scooters are morphologically distinct step-through two-wheelers. Merging with motorcycle violates project specification and introduces intra-class feature confusion. |
| `mini-bus_Others` | 315 | **EXCLUDED** | Excluded to preserve clear transit Bus feature representation and avoid ambiguous border-case vehicle sizes. |

---

## 6. Dataset Artifacts Generated

The converted dataset and audit artifacts reside at:
- Converted YOLO dataset: `D:/VisionToll/data/processed/vision_toll_yolo/`
  - `images/{train, val, test}`
  - `labels/{train, val, test}`
  - `data.yaml`
- Statistical audit report: `D:/VisionToll/results/dataset_audit/audit_summary.json`
- Full box-level metrics: `D:/VisionToll/results/dataset_audit/bounding_box_details.csv`
- Distribution plots: `results/dataset_audit/class_distribution.png`, `scale_distribution.png`, `objects_per_image.png`
- Visual ground truth previews: `results/dataset_audit/visual_samples/` (24 sample images)
- Complete composite contact sheet: `results/dataset_audit/contact_sheet.jpg`