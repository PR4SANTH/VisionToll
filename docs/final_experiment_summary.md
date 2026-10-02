# VisionToll: Final Multi-Model Experiment Summary

**Academic Context**: 23CSE473 Neural Networks and Deep Learning (Group A12)  
**Project**: VisionToll: Deep Learning-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring  
**Document Status**: Final Consolidated Academic Summary  

---

## 1. Dataset Overview

The VisionToll dataset was derived and curated from the First-Generation Vehicle Dataset (FGVD) to address automated vehicle classification and toll-lane surveillance requirements. The source imagery was audited for label integrity, bounding box coordinates, and visual quality. Annotations were converted into standardized format coordinates, with bounding boxes validated to eliminate degenerate dimensions, out-of-boundary coordinates, or duplicate records.

---

## 2. Five Target Classes (Taxonomy Definition)

The evaluation taxonomy is strictly restricted to five mutually exclusive target vehicle classes representative of Indian National Highway toll schedules:

1. **`Class 0: Bus`** — Multi-axle commercial passenger transport buses.
2. **`Class 1: Car`** — Standard passenger sedans, hatchbacks, SUVs, and personal motor vehicles.
3. **`Class 2: Motorcycle`** — Two-wheeled motorized transport (motorcycles, mopeds).
4. **`Class 3: Auto Rickshaw`** — Three-wheeled commercial passenger transit vehicles.
5. **`Class 4: Truck`** — Commercial freight carriers, lorries, and heavy-goods vehicles.

**Excluded / Quarantined Classes**:  
Classes outside the 5-class taxonomy present in raw data sources (including *Van*, *Pickup*, *Scooter*, and *Mini-bus*) were intentionally quarantined and excluded during preprocessing to prevent category ambiguity and preserve label consistency.

---

## 3. Dataset Partitioning & Split Isolation

The dataset was partitioned into fixed, non-overlapping splits adhering to standard machine learning protocols:

| Split Name | Image Count | Label File Count | Ground-Truth Instances | Primary Role in Study | Quarantine / Access Policy |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Train Split** | 3,535 | 3,535 | 12,388 | Model parameter optimization | Accessible for model training |
| **Validation Split** | 884 | 884 | 3,100 | Checkpoint selection & comparative benchmarking | Accessible for validation only |
| **Test Split** | 1,083 | 1,083 | 3,926 | Final held-out evaluation of selected deployment model | **STRICTLY LOCKED during exploration** |

Data integrity checks verified zero mutual image stems between the three partitions ($|\text{Train} \cap \text{Val}| = 0$, $|\text{Train} \cap \text{Test}| = 0$, $|\text{Val} \cap \text{Test}| = 0$).

---

## 4. Evaluated Detector Architectures

To explore the design space of automated vehicle detection under real-world toll camera constraints, six distinct object detection architectures representing diverse architectural paradigms were systematically evaluated:

1. **ARCH-01 (YOLOv8n)**: Lightweight single-stage anchor-free convolutional detector utilizing CSPDarknet backbone, C2f feature abstraction modules, and decoupled detection heads (3.0M parameters).
2. **ARCH-02 (YOLOv8s)**: Capacity-scaled single-stage anchor-free detector with increased channel dimensions and depth across the backbone and path aggregation network (11.1M parameters).
3. **ARCH-03 (YOLOv5s)**: Established single-stage anchor-based convolutional detector employing CSPNet backbone with predefined k-means anchor priors (7.0M parameters).
4. **ARCH-04 (Faster R-CNN ResNet50-FPN)**: Two-stage detector utilizing a ResNet-50 backbone, Feature Pyramid Network (FPN), Region Proposal Network (RPN), and RoI Align classification/regression subnetworks (41.4M parameters).
5. **ARCH-05 (SSD-Lite MobileNetV3 Large)**: Mobile-optimized single-stage detector utilizing inverted residual blocks and depthwise separable convolutions for edge efficiency (2.26M parameters).
6. **ARCH-06 (RetinaNet ResNet50-FPN)**: Dense anchor single-stage detector combining ResNet-50-FPN feature pyramids with Focal Loss ($\alpha=0.25, \gamma=2.0$) to counteract extreme foreground-background class imbalance (32.3M parameters).

*Note: An exploratory custom architecture (VisionTollNet) was investigated during intermediate experimentation; however, forensic analysis revealed severe feature-level assignment starvation in Stage 1, leading to its formal abandonment. It is excluded from the final six-model set.*

---

## 5. Training Protocols

All experiments were executed on an NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, compute capability 8.6) under fixed random seed (`seed=42`) for reproducibility:

| Architecture ID | Model | Input Resolution | Batch Size | Optimizer | Base LR | Scheduler | Epochs | Mixed Precision |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **ARCH-01** | YOLOv8n | $640 \times 640$ | 16 | AdamW | 0.002 | Linear warmup + Cosine | 50 | FP16 (AMP) |
| **ARCH-02** | YOLOv8s | $640 \times 640$ | 16 | AdamW | 0.002 | Linear warmup + Cosine | 50 | FP16 (AMP) |
| **ARCH-03** | YOLOv5s | $640 \times 640$ | 8 | SGD | 0.010 | Linear warmup + Linear | 50 | FP16 (AMP) |
| **ARCH-04** | Faster R-CNN | $640 \times 640$ | 4 | SGD | 0.005 | Fixed step decay | 30* | FP16 (AMP) |
| **ARCH-05** | SSD-Lite | $320 \times 320$ | 4 | AdamW | 0.001 | StepLR ($\gamma=0.5$ / 15 ep) | 16 | FP32 |
| **ARCH-06** | RetinaNet | $640 \times 640$ | 4 | SGD | 0.005 | Cosine Annealing | 50 | FP16 (AMP) |

*\*Faster R-CNN training was halted at epoch 37 due to compute resource bounds; the best intermediate checkpoint (epoch 30) was used for evaluation.*

---

## 6. Empirical Validation Results

The six architectures were evaluated on the held-out validation split (884 images) using their respective best validation checkpoints:

| Architecture ID | Model | Parameters | Best Epoch | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARCH-01** | YOLOv8n | 3,006,623 | 50 | 0.8579 | 0.7819 | 0.8181 | 0.8655 | 0.7197 |
| **ARCH-02** | YOLOv8s | 11,137,535 | 48 | 0.8289 | 0.8127 | 0.8207 | 0.8775 | 0.7399 |
| **ARCH-03** | YOLOv5s | 7,033,114 | 48 | 0.8341 | 0.8143 | 0.8241 | 0.8653 | 0.7163 |
| **ARCH-04** | Faster R-CNN | 41,372,781 | 30 | 0.6673 | 0.8445 | 0.7456 | 0.7850 | 0.6097 |
| **ARCH-05** | SSD-Lite | 2,261,960 | 16 | 0.7484 | 0.6535 | 0.6978 | 0.7157 | 0.4944 |
| **ARCH-06** | RetinaNet | 32,284,049 | 27 | 0.6557 | 0.8513 | 0.7408 | 0.7998 | 0.6408 |

### Observations:
- **YOLOv8s (ARCH-02)** recorded the highest overall validation mAP@0.50 (0.8775) and mAP@0.50:0.95 (0.7399) while maintaining a balanced precision (0.8289) and recall (0.8127).
- **YOLOv5s (ARCH-03)** exhibited competitive performance (0.8653 mAP@0.50, 0.7163 mAP@0.50:0.95) with a high F1-score (0.8241).
- **Faster R-CNN (ARCH-04)** and **RetinaNet (ARCH-06)** achieved high raw recall (0.8445 and 0.8513, respectively), but registered lower precision (0.6673 and 0.6557), reflecting higher false-positive proposal rates under dense traffic.
- **SSD-Lite (ARCH-05)** provided a compact model size (2.26M parameters) but experienced lower accuracy (0.4944 mAP@0.50:0.95), constrained by its lower spatial input resolution ($320 \times 320$).

---

## 7. Computational & Efficiency Profiles

Hardware execution efficiency was benchmarked on the target edge GPU (NVIDIA RTX 3050 Laptop):

| Architecture ID | Model | Parameters | Training Duration | Latency (ms) | Throughput (FPS) | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **ARCH-01** | YOLOv8n | 3.01M | 262.03 min | 5.0 ms | 200.0 FPS | ~2,100 MB |
| **ARCH-02** | YOLOv8s | 11.14M | 92.16 min | 7.9 ms | 126.6 FPS | ~4,034 MB |
| **ARCH-03** | YOLOv5s | 7.03M | 122.84 min | 6.2 ms | 161.3 FPS | ~2,250 MB |
| **ARCH-04** | Faster R-CNN | 41.37M | N/A (halted) | ~50.0 ms | ~20.0 FPS | ~2,045 MB |
| **ARCH-05** | SSD-Lite | 2.26M | 50.56 min | 11.3 ms | 88.4 FPS | ~410 MB |
| **ARCH-06** | RetinaNet | 32.28M | 856.65 min | 40.2 ms | 24.9 FPS | ~1,965 MB |

---

## 8. Motorcycle-Specific Diagnostic Results

The `Motorcycle` category (Class 2, 780 ground-truth instances in validation) represented the primary localization and occlusion challenge due to small bounding box areas and dense lane-sharing:

| Architecture ID | Model | Moto Precision | Moto Recall | Moto F1 | Moto mAP@0.50 | Moto mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **ARCH-01** | YOLOv8n | 0.8118 | 0.7634 | 0.7869 | 0.8390 | 0.5995 |
| **ARCH-02** | YOLOv8s | 0.7576 | 0.8115 | 0.7836 | 0.8397 | 0.6216 |
| **ARCH-03** | YOLOv5s | N/A | N/A | N/A | 0.8328 | 0.6005 |
| **ARCH-04** | Faster R-CNN | 0.6192 | 0.7859 | 0.6927 | N/A | N/A |
| **ARCH-05** | SSD-Lite | 0.6203 | 0.5487 | 0.5823 | N/A | 0.3186 |
| **ARCH-06** | RetinaNet | 0.6161 | 0.8128 | 0.7008 | N/A | 0.5111 |

- **YOLOv8s** achieved the highest strict localization quality on motorcycles (0.6216 mAP@0.50:0.95), resolving the sub-0.60 bottleneck observed in YOLOv8n and YOLOv5s.
- **RetinaNet** achieved high motorcycle recall (0.8128, 634/780 detected), on par with YOLOv8s (0.8115), but with lower precision (0.6161) resulting from dense background anchors.

---

## 9. Small-Object Diagnostics

Where small-object diagnostics were explicitly tracked:
- **ARCH-01 (YOLOv8n)**: Achieved 42.22% recall (19/45) on sub-$32\times32$ px distant motorcycles.
- **ARCH-02 (YOLOv8s)**: Achieved 33.33% recall (15/45) on sub-$32\times32$ px distant motorcycles, reflecting that spatial resolution ($640 \times 640$) rather than parameter count bounds extreme small-scale sensitivity.
- **ARCH-06 (RetinaNet)**: Evaluated at 29.01% recall on validation objects with area $<1,024 \text{ px}^2$, constrained by the absence of a high-resolution P2 feature level (lowest FPN stride is 8 at P3).

---

## 10. Test-Set Protocol & Quarantine Compliance

To prevent data contamination and circular model selection:
1. **Model selection and comparative analysis were conducted strictly using the validation split.**
2. **The held-out test set (1,083 images, 3,926 vehicle instances) remained locked and was not observed during any training, hyperparameter adjustment, or comparison procedure.**
3. **Only after formal model selection was frozen** was the final candidate (YOLOv8s) evaluated on the held-out test split, as documented in [`results/final_test/final_test_report.md`](file:///D:/VisionToll/results/final_test/final_test_report.md), achieving 0.8654 mAP@0.50 and 0.7335 mAP@0.50:0.95.

---

## 11. Final Selected Deployment Model

- **Selected Candidate**: **YOLOv8s (ARCH-02)**
- **Authoritative Checkpoint**: [`models/exp3_yolov8s/best.pt`](file:///D:/VisionToll/models/exp3_yolov8s/best.pt)
- **Verified SHA-256 Digest**:
  `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD`
- **Selection Basis**:
  - Highest validation mAP@0.50:0.95 (0.7399) and mAP@0.50 (0.8775).
  - Superior balance of precision (0.8289) and recall (0.8127).
  - Solid motorcycle localization breakthrough (0.6216 mAP@0.50:0.95).
  - High real-time inference throughput (126.6 FPS) suitable for edge toll monitoring.

---

## 12. Methodological Limitations

1. **Fixed Hardware Context**: All throughput and latency metrics were obtained on an NVIDIA RTX 3050 Laptop GPU; deployment on specialized edge SoCs (e.g., NVIDIA Jetson Orin) may exhibit distinct thermal and memory characteristics.
2. **Fixed Input Resolution**: Detectors were evaluated predominantly at $640 \times 640$ (and $320 \times 320$ for SSD-Lite); multi-scale testing or higher input resolutions ($1280 \times 1280$) were not explored due to VRAM bounds.
3. **Single-Seed Optimization**: Experiments utilized deterministic seed 42. Multi-seed confidence intervals were not generated due to training budget constraints.
4. **Scope Restriction**: The deployment application is strictly an image-based vehicle detection and census audit tool; multi-object video tracking (MOT), speed estimation, and automated number plate recognition (ANPR) were not implemented as part of this scope.

---

## 13. Reproducibility Information

- **Repository Root**: `D:/VisionToll`
- **Environment**: Python 3.12, PyTorch 2.6.0+cu124, torchvision 0.21.0, Ultralytics 8.3.9
- **Dataset Configuration**: [`data/processed/vision_toll_yolo/data.yaml`](file:///D:/VisionToll/data/processed/vision_toll_yolo/data.yaml)
- **Model Checkpoints**: Preserved under `models/` and `results/experiments/`
- **Evaluation Records**: Recorded in `results/final_model_comparison.csv` and `results/final_model_comparison.md`
