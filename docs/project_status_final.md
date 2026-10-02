# VisionToll: Final Project Status & Executive Sign-Off

**Project Title**: VisionToll: Deep Learning-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring  
**Academic Context**: 23CSE473 Neural Networks and Deep Learning (Group A12)  
**Status**: PROJECT COMPLETED — MODEL FROZEN — CONSOLIDATION FINALIZED  
**Date**: October 2026  

---

## 1. Project Overview & Deliverables Status

| Component | Status | Summary / Authoritative Artifact |
| :--- | :---: | :--- |
| **Dataset Engineering** | **FINALIZED** | 5,502 curated images partitioned into 3,535 train, 884 val, and 1,083 locked test images. 5 target classes strictly enforced. |
| **Six Detector Architectures** | **COMPLETED** | Six distinct architectures evaluated across 4 detector families (YOLOv8n, YOLOv8s, YOLOv5s, Faster R-CNN, SSD-Lite, RetinaNet). |
| **Validation Comparison** | **COMPLETED** | Rigorous empirical evaluation across all six architectures on the 884-image validation split (`results/final_model_comparison.csv` and `.md`). |
| **Model Selection & Freeze** | **FROZEN** | **YOLOv8s** (Experiment 3) confirmed as champion deployment model. Checkpoint frozen with SHA-256 digest in `models/FINAL_MODEL.txt`. |
| **Held-Out Test Protocol** | **PRESERVED** | Test set was strictly isolated and locked during all training and comparative selection phases; single final test evaluation documented in `results/final_test/final_test_report.md`. |
| **Interactive Application** | **OPERATIONAL** | Streamlit web application operational for static image vehicle census auditing (`app/app.py`). |
| **Scope Boundaries** | **DELIMITED** | Video multi-object tracking (MOT) and speed estimation are explicitly **not part of the final implemented system**. |
| **Exploratory Branches** | **CONCLUDED** | VisionTollNet custom architecture encountered feature starvation in Stage 1 and was formally abandoned; excluded from final comparison. |

---

## 2. Definitive Dataset Taxonomy

The final dataset enforces exactly five vehicle categories aligned with Indian National Highway toll classification standards:

- **`Class 0: Bus`**
- **`Class 1: Car`**
- **`Class 2: Motorcycle`**
- **`Class 3: Auto Rickshaw`**
- **`Class 4: Truck`**

**Excluded Categories**:  
All raw FGVD categories outside this taxonomy (*Van*, *Pickup*, *Scooter*, *Mini-bus*) were permanently quarantined during data preparation and are not recognized or predicted by the pipeline.

---

## 3. Final Six-Model Architecture Set

| Canonical ID | Detector Architecture | Family / Paradigm | Parameters | Key Empirical Characteristic |
| :---: | :--- | :--- | :---: | :--- |
| **ARCH-01** | YOLOv8n | Anchor-free single-stage CNN | 3.01M | Ultra-fast baseline (200 FPS); solid Car/Auto Rickshaw detection |
| **ARCH-02** | YOLOv8s | Anchor-free single-stage CNN (Capacity) | 11.14M | **Selected Champion**: highest mAP50-95 (0.7399), highest mAP50 (0.8775), breaks motorcycle bottleneck (0.6216 mAP50-95) |
| **ARCH-03** | YOLOv5s | Anchor-based single-stage CNN | 7.03M | Strong competitive baseline (0.7163 mAP50-95, 0.8241 F1) |
| **ARCH-04** | Faster R-CNN ResNet50-FPN | Two-stage proposal-based detector | 41.37M | High recall (0.8445), but heavy computation (~50 ms latency) and lower precision (0.6673) |
| **ARCH-05** | SSD-Lite MobileNetV3 Large | Mobile single-stage detector | 2.26M | Minimal footprint (2.26M parameters, 11.3 ms latency); lower accuracy (0.4944 mAP50-95) |
| **ARCH-06** | RetinaNet ResNet50-FPN | Dense anchor single-stage with Focal Loss | 32.28M | Highest raw recall (0.8513) and strong motorcycle recall (0.8128); 24.9 FPS throughput |

---

## 4. Frozen Production Model Specifications

- **Candidate Model**: **YOLOv8s** (Experiment 3 / ARCH-02)
- **Deployment Checkpoint**: `D:/VisionToll/models/exp3_yolov8s/best.pt`
- **Verified SHA-256 Digest**:
  ```
  5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD
  ```
- **Operational Metrics (Validation Split, 884 images)**:
  - Precision: **0.8289**
  - Recall: **0.8127**
  - F1-Score: **0.8207**
  - mAP@0.50: **0.8775**
  - mAP@0.50:0.95: **0.7399**
  - Motorcycle Recall: **0.8115**
  - Motorcycle mAP@0.50:0.95: **0.6216**
  - Throughput: **126.6 FPS** (7.9 ms pipeline latency on NVIDIA RTX 3050 Laptop GPU)
- **Held-Out Test Set Verification (1,083 images)**:
  - Test mAP@0.50: **0.8654**
  - Test mAP@0.50:0.95: **0.7335**
  - Generalization Gap: **-0.64 pp** relative to validation mAP50-95, verifying strong real-world generalization without overfitting.

---

## 5. Scope Delimitations

1. **Application Interface**: The deployed system interface (`app/app.py`) is an interactive Streamlit application designed for **static image upload, multi-class vehicle detection, and toll census auditing**.
2. **Video & Tracking Delimitation**: Video-based multi-object tracking (e.g., DeepSORT, ByteTrack), trajectory analysis, speed measurement, and automated license plate recognition (ALPR/ANPR) are **explicitly out of scope** and were not implemented in the final production release.
3. **VisionTollNet Retirement**: The custom VisionTollNet architecture reached a definitive technical boundary in Stage 1 and has been formally retired. No further development or training of VisionTollNet will occur.

---

## 6. Project Documentation Sitemap

- Final Comparison Table (CSV): [`results/final_model_comparison.csv`](file:///D:/VisionToll/results/final_model_comparison.csv)
- Final Comparison Report (Markdown): [`results/final_model_comparison.md`](file:///D:/VisionToll/results/final_model_comparison.md)
- Architectural Mapping Documentation: [`docs/final_model_comparison.md`](file:///D:/VisionToll/docs/final_model_comparison.md)
- Consolidated Experiment Summary: [`docs/final_experiment_summary.md`](file:///D:/VisionToll/docs/final_experiment_summary.md)
- Final Test Evaluation Report: [`results/final_test/final_test_report.md`](file:///D:/VisionToll/results/final_test/final_test_report.md)
- Streamlit Web Dashboard: [`app/app.py`](file:///D:/VisionToll/app/app.py)
