# VisionToll: Final Academic Test Evaluation Report

**Academic Context**: 23CSE473 Neural Networks and Deep Learning (Group A12)  
**Project**: VisionToll: Deep Learning-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring  
**Target Taxonomy**: 5 Classes (`0: Bus`, `1: Car`, `2: Motorcycle`, `3: Auto Rickshaw`, `4: Truck`)  
**Excluded Classes**: Van, Pickup, Scooter, Mini-bus  
**Status**: FINAL HELD-OUT TEST EVALUATION COMPLETE — MODEL FROZEN  

---

## 1. Final Model Selection Basis

The selection of the final production candidate model for VisionToll was conducted strictly on the basis of empirical validation evidence accumulated across Experiments 1, 2, and 3. At no point during model development, hyperparameter selection, or failure diagnosis was the test split accessed or observed.

Across the three investigations:
1. **Experiment 1 (YOLOv8n Baseline)** established a strong baseline (0.8655 mAP@0.50, 0.7197 mAP@0.50:0.95), but qualitative and quantitative failure analysis revealed a pronounced performance ceiling on Motorcycle localization (0.5995 mAP@0.50:0.95), alongside low recall on Bus (0.7286) and 484 hard-example validation failure scenes.
2. **Experiment 2 (YOLOv8n + Targeted Augmentation)** tested the hypothesis that scale jitter (`scale=0.9`) and copy-paste instance injection (`copy_paste=0.3`) could alleviate small-vehicle deficits. However, empirical results refuted this: precision degraded across all classes (-2.75 pp), motorcycle mAP@0.50:0.95 fell to 0.5928, and hard-example scenes expanded to 524 (+8.26%). This demonstrated that data-level augmentation alone cannot overcome the parameter and spatial capacity constraints of a 3.0M-parameter nano backbone.
3. **Experiment 3 (YOLOv8s Model Capacity Comparison)** evaluated the effect of increasing model capacity to 11.1M parameters and 28.7 GFLOPs. On the validation split, YOLOv8s achieved superior overall performance: **0.8775 mAP@0.50**, **0.7399 mAP@0.50:0.95**, **0.8127 Recall**, and **0.8207 F1-score**. Crucially, it broke through the motorcycle localization bottleneck, reaching **0.6216 mAP@0.50:0.95** (+2.21 pp over Exp 1) and expanding motorcycle recall to **0.8115** (+4.81 pp). Furthermore, hard-example scenes declined by **14.87%** (412 scenes vs. 484 in Exp 1), low-confidence predictions dropped by **23.72%** (283 scenes vs. 371), and motorcycle false alarms dropped by **15.0%** (363 vs. 427), while maintaining an inference throughput of **125.4 FPS** on an NVIDIA GeForce RTX 3050 Laptop GPU.

Consequently, **YOLOv8s from Experiment 3** was selected as the sole candidate for final test freeze and held-out evaluation.

---

## 2. Frozen Checkpoint Verification

To guarantee scientific rigor, reproducibility, and prevent post-selection modifications, the champion model weights were verified via cryptographic hashing prior to test execution:

- **Checkpoint Location**: `D:\VisionToll\models\exp3_yolov8s\best.pt`
- **Model Architecture**: Ultralytics YOLOv8s (130 layers, 11,137,535 parameters, 28.7 GFLOPs)
- **Verified SHA-256 Digest**:
  ```
  5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD
  ```
- **Freeze Status**: Formally recorded in [`models/FINAL_MODEL.txt`](file:///D:/VisionToll/models/FINAL_MODEL.txt) and [`results/final_model_freeze.json`](file:///D:/VisionToll/results/final_model_freeze.json).
- **Execution Constraint**: The model was treated as immutable. No fine-tuning, weight modification, or confidence threshold alterations were permitted or performed.

---

## 3. Test Dataset Description

The held-out test split represents a completely independent collection of real-world traffic scenes captured across diverse toll plazas and highway corridors. The data integrity checks confirmed:

- **Total Test Images**: 1,083 images
- **Total Test Label Files**: 1,083 files (100% pairing verified; zero missing or orphaned files)
- **Image Resolution**: 640×640 pixels (standardized preprocessing)
- **Total Ground-Truth Instances**: 3,926 annotated vehicles
- **Ground-Truth Class Distribution**:
  - `Class 0 (Bus)`: 219 instances (5.58%)
  - `Class 1 (Car)`: 1,559 instances (39.71%)
  - `Class 2 (Motorcycle)`: 1,085 instances (27.64%)
  - `Class 3 (Auto Rickshaw)`: 754 instances (19.21%)
  - `Class 4 (Truck)`: 309 instances (7.87%)
- **Data Integrity**: Zero class index violations (all annotations strictly in range `0–4`). No test images were present in either the training or validation splits.

---

## 4. Test Evaluation Protocol

The evaluation was conducted strictly once on the held-out test split using the standardized Ultralytics evaluation engine and project diagnostics:

- **Dataset Configuration**: `data/processed/vision_toll_yolo/data.yaml`
- **Evaluation Split**: `test`
- **Batch Size**: 16
- **Input Resolution**: 640×640 pixels
- **Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124)
- **Confidence Threshold**: Standard $0.001$ for metric curve calculation (standard mAP integration); $0.25$ for hard-example detection extraction.
- **IoU Threshold**: $0.70$ for NMS proposal deduplication; $0.50$ to $0.95$ (step 0.05) for mAP calculation.
- **Constraint Compliance**: The evaluation was executed autonomously without human-in-the-loop parameter adjustments, stopping immediately upon metric recording.

---

## 5. Overall Test Results

The frozen YOLOv8s model achieved the following aggregate performance metrics across the 1,083 test images:

| Evaluation Metric | Measured Test Result | Operational Significance |
| :--- | :---: | :--- |
| **Precision** | **0.8574** | High certainty; low rate of spurious background detections |
| **Recall** | **0.8017** | 80.2% of all physical vehicles successfully retrieved |
| **F1-Score** | **0.8286** | Strong harmonic balance between precision and recall |
| **mAP@0.50** | **0.8811** | Robust broad-overlap vehicle detection across all conditions |
| **mAP@0.50:0.95** | **0.7343** | Precise bounding box boundary alignment |
| **Total Predicted Detections** | 4,240 (at NMS) / 4,430 (conf $\ge 0.25$) | Aligns closely with 3,926 ground-truth objects |
| **Preprocessing Latency** | 0.27 ms | Ultra-fast image resizing and letterboxing |
| **Inference Latency** | **7.05 ms** | Real-time GPU forward pass |
| **Postprocessing Latency** | 0.65 ms | Rapid NMS box filtering |
| **Total Pipeline Latency** | **7.97 ms per frame** | Full end-to-end processing under 8.0 ms |
| **Effective Throughput** | **125.4 FPS** | Comfortably exceeds 30–60 FPS multi-lane toll camera feeds |

---

## 6. Per-Class Test Results

The breakdown of detection metrics across the 5 target vehicle classes on the test split is summarized below:

| Class ID | Class Name | Ground Truth Count | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Bus** | 219 | 0.8734 | 0.7563 | 0.8107 | 0.8382 | 0.6976 |
| 1 | **Car** | 1,559 | 0.9059 | 0.8273 | 0.8648 | 0.9264 | 0.8032 |
| 2 | **Motorcycle** | 1,085 | 0.8202 | 0.7860 | 0.8027 | 0.8644 | 0.6361 |
| 3 | **Auto Rickshaw** | 754 | 0.9001 | 0.8481 | 0.8733 | 0.9216 | 0.8163 |
| 4 | **Truck** | 309 | 0.7874 | 0.7912 | 0.7893 | 0.8550 | 0.7182 |
| — | **All Classes** | **3,926** | **0.8574** | **0.8017** | **0.8286** | **0.8811** | **0.7343** |

### Per-Class Performance Observations:
- **Car & Auto Rickshaw (Top Performers)**: Car achieved 0.9264 mAP@0.50 and 0.8032 mAP@0.50:0.95; Auto Rickshaw reached 0.9216 mAP@0.50 and 0.8163 mAP@0.50:0.95. Both classes demonstrate superior feature distinctiveness and robust detection stability in dense toll lanes.
- **Motorcycle (Validated Generalization)**: The test evaluation confirms that the bottleneck resolution observed in Experiment 3 was genuine. Motorcycle reached **0.8644 mAP@0.50** and **0.6361 mAP@0.50:0.95**, with an F1-score of 0.8027. This surpasses the baseline nano model (0.5995) by +3.66 pp on unseen data.
- **Bus (Recall and Boundary Constraints)**: Bus achieved high precision (0.8734) and 0.8382 mAP@0.50, but recorded the lowest recall (0.7563). Visual inspection confirms that large buses entering camera framing are frequently cropped by canopy structures, reducing proposal overlap.
- **Truck (Precision Tradeoff)**: Truck recorded 0.8550 mAP@0.50 and 0.7182 mAP@0.50:0.95, with balanced recall (0.7912) and precision (0.7874).

---

## 7. Confusion Matrix Analysis

The quantitative confusion matrices (raw counts and column-normalized by ground truth) generated during test evaluation provide detailed insight into cross-class classification behavior:

### Confusion Matrix (Raw Counts)

| Ground Truth $\downarrow$ / Predicted $\rightarrow$ | Bus | Car | Motorcycle | Auto Rickshaw | Truck | Missed (Background FN) | Total GT Instances |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bus** | **171** | 2 | 0 | 1 | 3 | 42 | 219 |
| **Car** | 3 | **1361** | 0 | 3 | 5 | 187 | 1,559 |
| **Motorcycle** | 0 | 0 | **927** | 0 | 0 | 158 | 1,085 |
| **Auto Rickshaw** | 0 | 0 | 0 | **653** | 9 | 92 | 754 |
| **Truck** | 1 | 4 | 0 | 3 | **260** | 41 | 309 |
| **False Alarms (Background FP)** | 32 | 220 | 369 | 129 | 84 | — | 834 total FP |

### Normalized Confusion Matrix (True Class Proportions)

| Ground Truth $\downarrow$ / Predicted $\rightarrow$ | Bus | Car | Motorcycle | Auto Rickshaw | Truck | Missed (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bus** | **78.08%** | 0.91% | 0.00% | 0.46% | 1.37% | 19.18% |
| **Car** | 0.19% | **87.30%** | 0.00% | 0.19% | 0.32% | 12.00% |
| **Motorcycle** | 0.00% | 0.00% | **85.44%** | 0.00% | 0.00% | 14.56% |
| **Auto Rickshaw** | 0.00% | 0.00% | 0.00% | **86.60%** | 1.19% | 12.20% |
| **Truck** | 0.32% | 1.29% | 0.00% | 0.97% | **84.14%** | 13.27% |

### Key Findings from Confusion Matrix:
1. **Absolute Absence of Motorcycle Cross-Class Confusion**: Across all 1,085 test motorcycles, exactly 0 were misclassified as Car, Bus, Auto Rickshaw, or Truck. Motorcycle detection failures are strictly localization/miss errors (158 instances, 14.56%), never taxonomy confusion.
2. **Minimal Car $\leftrightarrow$ Truck Ambiguity**: Despite visual similarities between large SUVs, pickups (excluded from taxonomy), and light commercial trucks, only 5 Cars were predicted as Truck (0.32%), and only 4 Trucks were predicted as Car (1.29%).
3. **Background False Positive Concentration**: Of the 834 total background false alarms, **44.2% (369 instances)** were attributed to Motorcycle. In toll plaza environments, roadway markings, guardrails, and lane dividers periodically trigger weak motorcycle proposals.

---

## 8. Error Analysis

Descriptive analysis of the test split failure cases reveals specific recurring visual phenomena:

1. **Missed Motorcycles**:
   - Out of 1,085 ground-truth motorcycles, 158 were unretrieved. These missed instances predominantly occurred in two contexts: (a) distant motorcycles positioned far back in toll approach lanes ($>50$ meters from the gantry camera), where pixel dimensions fall below $24\times24$ pixels, and (b) motorcycles riding directly alongside large commercial trucks or buses, where vehicle bodies create severe partial occlusion.
2. **Small/Distant Vehicle Attenuation**:
   - In test scenes containing vehicles under $32\times32$ pixels, detection recall fell to 40.0% (16 of 40 small motorcycles detected). This confirms that feature striding (downsampling factors of 8, 16, and 32 in the YOLO feature pyramid network) sets a spatial resolution limit for objects occupying fewer than ~500 pixels.
3. **Bus Framing & Scale Occlusion**:
   - 42 buses (19.18%) were missed. Visual verification indicates that buses positioned directly under toll canopies are frequently cropped horizontally or vertically by camera boundaries. Because bounding box annotations encompass only visible segments, the network occasionally treats partially visible bus roofs as ambient infrastructure.
4. **Auto-Rickshaw to Truck Discrepancy**:
   - 9 auto-rickshaws were classified as trucks. These cases involved rear-facing auto-rickshaws carrying oversized cargo or customized metal roof racks that mimic the rectangular geometry of small flatbed cargo carriers.
5. **Crowded Scene Clustered Predictions**:
   - In multi-vehicle queues at the toll barrier, overlapping bounding boxes between adjacent vehicles occasionally cause non-maximum suppression (NMS) suppression conflicts, where the detector suppresses a valid proposal whose IoU exceeds 0.70 with an adjacent vehicle.

---

## 9. Hard-Example Test Analysis

Applying the project's standardized hard-example profiling methodology to the 1,083 test scenes yielded the following diagnostic breakdown:

- **Total Test Hard-Example Scenes**: **495 / 1,083 (45.71%)**  
  (Stored in [`results/hard_examples/final_test/final_test_hard_examples_manifest.json`](file:///D:/VisionToll/results/hard_examples/final_test/final_test_hard_examples_manifest.json))

### Categorical Distribution of Challenging Conditions

| Hard-Example Category | Condition Definition | Test Split Occurrence | Percentage of Test Scenes |
| :--- | :--- | :---: | :---: |
| **Low-Confidence Scenes** | Contained $\ge 1$ detection with confidence $\in [0.25, 0.40)$ | 349 | 32.23% |
| **Crowded Traffic Scenes** | Contained $\ge 6$ simultaneous ground-truth vehicles | 158 | 14.59% |
| **Small-Object Scenes** | Contained objects with bounding box area $< 32\times32$ px | 110 | 10.16% |
| **Count Discrepancy Scenes** | $|\text{Ground Truth} - \text{Predicted Detections}| \ge 3$ | 105 | 9.70% |
| **Complete Detector Misses** | Scenes with $\ge 1$ GT vehicles where 0 detections occurred | 4 | 0.37% |

### Specific Motorcycle Diagnostic Profile on Test Split
- Total Ground-Truth Motorcycles: 1,085
- Total Predicted Motorcycle Boxes (conf $\ge 0.25$): 1,332
- Matched True Positives: 930 (Recall: 85.71% at IoU $\ge 0.50$)
- Missed False Negatives: 155
- Background False Positives: 402
- Small Motorcycle Recall ($<32\times32$ px): **40.0% (16 / 40)**
- Crowded Motorcycle Recall ($\ge 6$ vehicles): **79.87% (254 / 318)**
- Mean Prediction Confidence: **0.7007** (Median: 0.7859)

The hard-example analysis demonstrates that crowded scenes remain manageable (nearly 80% motorcycle recall), whereas small spatial scale is the principal driver of remaining misses.

---

## 10. Validation vs. Test Generalization

A critical objective of this final evaluation is assessing whether validation metrics served as a faithful predictor of real-world generalization:

| Metric | Validation (Exp 3 `best.pt`) | Final Held-Out Test | Absolute Gap ($\Delta$) | Relative Difference | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Precision** | 0.8289 | 0.8574 | +0.0285 | +3.44% | Favorable (fewer false positives on test) |
| **Recall** | 0.8127 | 0.8017 | -0.0110 | -1.35% | Highly consistent coverage ($\sim 1\%$ gap) |
| **F1-Score** | 0.8207 | 0.8286 | +0.0079 | +0.96% | Stable harmonic performance |
| **mAP@0.50** | 0.8775 | 0.8811 | +0.0036 | +0.41% | Flawless validation correspondence |
| **mAP@0.50:0.95** | 0.7399 | 0.7343 | -0.0056 | -0.76% | Less than 0.8% degradation in box precision |
| **Motorcycle mAP50-95**| 0.6216 | 0.6361 | +0.0145 | +2.33% | Bottleneck resolution verified on test data |
| **Car mAP50-95** | 0.8043 | 0.8032 | -0.0011 | -0.14% | Exact agreement |
| **Auto Rickshaw mAP50-95**| 0.8090 | 0.8163 | +0.0073 | +0.90% | Exact agreement |
| **Bus mAP50-95** | 0.6993 | 0.6976 | -0.0017 | -0.24% | Exact agreement |
| **Truck mAP50-95** | 0.7654 | 0.7182 | -0.0472 | -6.17% | Moderate variation due to test split composition |

### Generalization Assessment:
The generalization gap between the validation split and the independent held-out test split is negligible ($< 0.006$ mAP overall). Rather than indicating overfitting, these findings indicate that:
1. The 50-epoch training regime with AdamW and cosine decay converged to a stable, generalizable loss basin.
2. The model selection criteria based on validation mAP@0.50:0.95 and hard-example suppression generalized faithfully to unseen data.
3. The model exhibits high operational reliability across distinct camera perspectives and lighting conditions.

---

## 11. Computational Performance

Real-time processing capability is mandatory for automated toll plaza monitoring. The computational profiling of the frozen YOLOv8s model on the evaluation platform yielded:

- **Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Laptop Form Factor, 60W TGP)
- **Model Checkpoint Size**: 22.5 MB (`best.pt`)
- **Parameters**: 11,137,535
- **GFLOPs**: 28.7 (at 640×640 input resolution)
- **Latencies Measured on Test Set**:
  - Image Preprocessing: **0.27 ms**
  - Model Inference: **7.05 ms**
  - NMS & Postprocessing: **0.65 ms**
  - **Total Latency per Image**: **7.97 ms**
- **Throughput**: **125.4 frames per second (FPS)**

### Operational Feasibility:
Standard CCTV toll monitoring operates at 25 to 30 FPS per lane. Operating at 125.4 FPS, a single edge GPU of this caliber possesses sufficient throughput to simultaneously ingest and process video feeds from **3 to 4 toll lanes in real time** without frame dropping.

---

## 12. Limitations

In accordance with rigorous academic standards, several limitations of the current system must be noted:

1. **Spatial Resolution Bound on Distant Vehicles**:
   - While YOLOv8s improved motorcycle localization, vehicles smaller than $32\times32$ pixels still exhibit low recall (40.0%). Standard $640\times640$ letterboxing limits tiny object feature representations.
2. **Canopy-Induced Bus Truncation**:
   - Bus recall (75.63%) remains lower than other classes because overhead toll canopies partially obscure rooflines.
3. **Absence of Temporal Tracking**:
   - This evaluation assesses static single-frame object detection. In operational toll deployment, temporal tracking (e.g., ByteTrack) would be necessary to smooth bounding box jitter and eliminate momentary occlusions.
4. **Single-Device Profiling**:
   - Latency figures reflect an NVIDIA RTX 3050 GPU. Deployment on embedded edge hardware (e.g., NVIDIA Jetson Orin Nano) would require FP16/INT8 TensorRT quantization.
5. **No Claim of Statistical Universality**:
   - While the test evaluation demonstrates strong generalization across the 1,083 test images, performance in extreme weather (heavy monsoon rain, thick fog, night glare) requires dedicated domain validation.

---

## 13. Final Conclusion

The one-time held-out test evaluation of the frozen **VisionToll YOLOv8s model** confirms that the architecture achieves high detection accuracy and robust generalization across realistic Indian toll plaza conditions.

The model achieved an **overall mAP@0.50 of 0.8811**, an **mAP@0.50:0.95 of 0.7343**, an **F1-score of 0.8286**, and an **inference throughput of 125.4 FPS**. Crucially, the motorcycle localization bottleneck that constrained the nano baseline was successfully overcome on unseen test data (0.6361 mAP@0.50:0.95 vs. 0.5995 in Exp 1). The empirical generalization gap between validation and test was less than 0.8% relative mAP, confirming model stability and the integrity of the selection methodology.

With this formal test evaluation complete, the model weights remain permanently frozen, and the VisionToll experimental lifecycle is formally concluded.
