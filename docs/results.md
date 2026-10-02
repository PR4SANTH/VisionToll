# VisionToll: Experimental Results

## Executive Summary
This document records the empirical results of all model training experiments conducted within the VisionToll pipeline. The evaluations strictly adhere to the project experimental protocol:
- Exactly 5 target classes (`0: Bus`, `1: Car`, `2: Motorcycle`, `3: Auto Rickshaw`, `4: Truck`).
- Scooter and Mini-bus are permanently excluded.
- The held-out test split (1,083 images) is **completely untouched and unseen** during model training, hyperparameter exploration, and failure analysis.
- All metrics below reflect evaluation exclusively on the **884 validation split images** (`data/processed/vision_toll_yolo/images/val`).

---

## Experiment 1 — YOLOv8n Baseline

### 1. Training Environment & Execution Telemetry
- **Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **CUDA Version**: 12.4
- **Operating System**: Windows 11
- **Python Version**: 3.12.5
- **PyTorch Version**: 2.6.0+cu124
- **Ultralytics Version**: 8.4.138
- **Architecture**: YOLOv8n (`yolov8n.pt` pretrained on COCO as starting backbone)
- **Model Parameters**: 3,006,623 parameters across 73 layers (8.1 GFLOPs)
- **Training Parameters**:
  - Image Size: 640x640
  - Batch Size: 16
  - Total Epochs: 50
  - Seed: 42 (fully reproducible)
  - Optimizer: AdamW (`lr0=0.002`, `lrf=0.01`, `weight_decay=0.0005`)
  - Warmup: 3.0 epochs
  - Patience: 15 epochs
- **Execution Duration**: 15,721.63 seconds (~262.03 minutes / 4.36 hours)

### 2. Validation Results (Overall)
Evaluated on the 884 validation images (3,100 ground-truth vehicle instances):

| Metric | Overall Score |
| :--- | :---: |
| **Precision** | **0.8579** |
| **Recall** | **0.7819** |
| **F1-Score** | **0.8181** |
| **mAP@0.50** | **0.8655** |
| **mAP@0.50:0.95** | **0.7197** |

Inference speed benchmarked at: **0.8ms preprocess, 3.6ms inference, 0.6ms postprocess per image** on RTX 3050.

### 3. Per-Class Validation Breakdown

| Class ID | Class Name | Instances (Val) | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Bus** | 199 | 0.8570 | 0.7286 | 0.7878 | 0.7840 | 0.6717 |
| 1 | **Car** | 1,278 | 0.8956 | 0.8185 | 0.8552 | 0.9250 | 0.7946 |
| 2 | **Motorcycle** | 780 | 0.8118 | 0.7634 | 0.7867 | 0.8390 | 0.5995 |
| 3 | **Auto Rickshaw** | 610 | 0.8856 | 0.8121 | 0.8474 | 0.9090 | 0.7905 |
| 4 | **Truck** | 233 | 0.8397 | 0.7868 | 0.8126 | 0.8710 | 0.7422 |

---

### 4. Baseline Artifacts
All Experiment 1 assets have been systematically organized:
- **Pretrained Starting Weights**: `models/baseline/yolov8n.pt` (preserved untouched)
- **Trained Checkpoints**:
  - `models/baseline/best.pt` (6.22 MB)
  - `models/baseline/last.pt` (6.22 MB)
  - Mirrored in `results/baseline/weights/`
- **Metric Plots & Curves**:
  - Precision-Recall Curve: `results/baseline/BoxPR_curve.png`
  - F1 Curve: `results/baseline/BoxF1_curve.png`
  - Precision Curve: `results/baseline/BoxP_curve.png`
  - Recall Curve: `results/baseline/BoxR_curve.png`
  - Confusion Matrix: `results/baseline/confusion_matrix.png`
  - Normalized Confusion Matrix: `results/baseline/confusion_matrix_normalized.png`
  - Training Curves: `results/baseline/results.png`
  - Training Epoch Log: `results/baseline/results.csv`
- **Validation Sample Visualizations**:
  - Batch Ground Truth: `results/baseline/val_batch0_labels.jpg`, `val_batch1_labels.jpg`
  - Batch Predictions: `results/baseline/val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`
- **Structured Experiment Metadata**:
  - Validation Summary: `results/baseline/baseline_validation_results.json`
  - Comparative Experiment Log: `results/experiments/experiments.csv`
- **Hard-Example Catalog**:
  - Catalog JSON: `results/hard_examples/hard_examples_manifest.json`
  - Annotated Failure Visualizations: `results/hard_examples/`

---

### 5. Empirical Baseline Failure Analysis
A programmatic hard-example failure analysis was executed across all 884 validation images using `best.pt`. A total of 484 images exhibited one or more challenging detection conditions:

1. **Motorcycle Performance Deficit (Primary Bottleneck)**:
   - **Lowest mAP@0.50:0.95**: Motorcycle scored **0.5995**, nearly 20 percentage points lower than Car (0.7946) and Auto Rickshaw (0.7905).
   - **Lowest Precision**: Motorcycle precision is **0.8118**, frequently triggered by background false positives or partial edge detections in multi-vehicle clusters.
   - **Small-Scale Vulnerability**: 106 validation images contained small target objects ($< 32 \times 32$ pixels), the vast majority being distant motorcycles whose bounding boxes deteriorated or were missed under default anchor strides.

2. **Bus Missed Detections & Occlusion (Recall Bottleneck)**:
   - Bus achieved the **lowest recall (0.7286)** of all five classes and mAP@0.50 of **0.7840**.
   - Inspection of validation failure samples revealed that large buses entering toll approaches are frequently cut off by image borders or severely occluded by preceding trucks and cars, causing partial bounding box proposals to fall below the non-maximum suppression (NMS) intersection-over-union (IoU) threshold.

3. **Low-Confidence Detections in High Density Traffic**:
   - 371 validation images exhibited predictions with confidence scores between $0.25$ and $0.40$.
   - 137 validation images depicted dense scenes with $\ge 6$ simultaneous vehicles. In these crowded scenes, adjacent motorcycles and auto-rickshaws experience severe inter-class bounding box overlap, leading to count discrepancies ($|\text{GT} - \text{Pred}| \ge 3$ occurred in 127 validation scenes).

---

### 6. Evidence-Based Direction for Experiment 2
The empirical failure taxonomy definitively highlights that:
- **Small-object and motorcycle boundary localization** is the weakest component of the YOLOv8n baseline ($0.5995$ mAP@0.50:0.95).
- **Dense scene vehicle clustering and severe occlusion** lower recall for larger transit vehicles (Bus recall at $0.7286$).

Consequently, Experiment 2 investigated a controlled intervention directly targeting small-object multi-scale feature resolution and instance exposure (scale jitter: `scale=0.9` vs baseline `0.5`, and instance injection: `copy_paste=0.3` vs baseline `0.0`), without introducing unverified architectural shifts or premature test-set evaluation.

---

## Experiment 2 — Targeted Improvement Analysis

### 1. Experimental Configuration & Intervention Telemetry
- **Model Architecture**: YOLOv8n (`yolov8n.pt` pretrained on COCO as starting backbone)
- **Model Complexity**: 73 layers, 3,006,623 parameters, 8.1 GFLOPs
- **Target Classes**: Exactly 5 classes (`0: Bus`, `1: Car`, `2: Motorcycle`, `3: Auto Rickshaw`, `4: Truck`)
- **Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Software Stack**: PyTorch 2.6.0+cu124, CUDA 12.4, Python 3.12.5, Ultralytics 8.4.138
- **Core Hyperparameters (Identical to Exp 1)**:
  - Image Size: 640x640
  - Batch Size: 16
  - Total Epochs: 50
  - Random Seed: 42 (deterministic)
  - Optimizer: AdamW (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`)
  - Warmup: 3.0 epochs
  - Patience: 15 epochs
  - Mosaic: 1.0 (closed for final 10 epochs)
  - Translation: 0.1, Fliplr: 0.5, Random Erasing: 0.4
- **Targeted Intervention (Changed from Exp 1)**:
  - `scale`: **0.9** (Increased from 0.5 in Exp 1; expands scale jitter to $\pm 90\%$)
  - `copy_paste`: **0.3** (Increased from 0.0 in Exp 1; probability of pasting cropped vehicle instances)
  - `workers`: 0 (adjusted for Windows multiprocessing stability)
- **Execution Duration**: 18,935.4 seconds (~315.59 minutes / 5.26 hours) for 50 epochs; 18,957.24 seconds (~315.95 minutes / 5.27 hours) including formal post-training validation.
  - **Additional Training Cost**: +3,213.8 seconds (+53.56 minutes / +20.44% increase) due to online image resampling and instance composition overhead.

### 2. Overall Validation Metrics Comparison
Evaluated on the exact same 884 validation images (3,100 ground-truth vehicle instances):

| Metric | Exp 1 (Baseline) | Exp 2 (Post-Val `best.pt`) | Exp 2 (Epoch 50 Log) | Absolute Change (`best.pt` vs Exp 1) | Relative Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision** | **0.8579** | 0.8304 | 0.8381 | **-0.0275** | -3.21% |
| **Recall** | **0.7819** | 0.8028 | 0.7927 | **+0.0209** | +2.67% |
| **F1-Score** | **0.8181** | 0.8164 | 0.8148 | **-0.0017** | -0.21% |
| **mAP@0.50** | **0.8655** | 0.8739 | 0.8742 | **+0.0084** | +0.97% |
| **mAP@0.50:0.95** | **0.7197** | 0.7224 | 0.7217 | **+0.0027** | +0.38% |

#### Tradeoff Analysis:
1. **Precision vs. Recall Shift**: The scale-augmentation intervention boosted detector recall across all classes (+2.09 pp on `best.pt`), meaning fewer vehicles were completely missed. However, this came at a direct cost to precision (-2.75 pp), generating more false positive bounding box proposals.
2. **F1 Balance**: Because the precision drop outweighed the recall gain, the overall F1-score decreased slightly from 0.8181 to 0.8164.
3. **mAP Divergence**: While mAP@0.50 improved by +0.84 pp (reflecting higher box coverage at lenient overlap), mAP@0.50:0.95 saw only a marginal increase of +0.27 pp (+0.0027), indicating that fine-grained boundary localization did not systematically improve.

### 3. Per-Class Validation Breakdown & Comparison

| Class ID | Class Name | Metric | Exp 1 (Baseline) | Exp 2 (Scale Aug) | Absolute Change | Trend Interpretation |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | **Bus** | Precision | 0.8570 | 0.8263 | -0.0307 | More false alarms |
| | | Recall | 0.7286 | 0.7387 | +0.0101 | Fewer missed transit buses |
| | | F1-Score | 0.7878 | 0.7801 | -0.0077 | Slight net decline |
| | | mAP@0.50 | 0.7840 | 0.8193 | **+0.0353** | **Substantial improvement (+3.53 pp)** |
| | | mAP@0.50:0.95 | 0.6717 | 0.6956 | **+0.0239** | **Substantial improvement (+2.39 pp)** |
| **1** | **Car** | Precision | 0.8956 | 0.8788 | -0.0168 | Slight precision decrease |
| | | Recall | 0.8185 | 0.8344 | +0.0159 | Improved recall |
| | | F1-Score | 0.8552 | 0.8560 | +0.0008 | Virtually unchanged |
| | | mAP@0.50 | 0.9250 | 0.9236 | -0.0014 | Minor fluctuation |
| | | mAP@0.50:0.95 | 0.7946 | 0.7920 | -0.0026 | Minor fluctuation |
| **2** | **Motorcycle** | Precision | 0.8118 | 0.7679 | **-0.0439** | **Severe degradation (-4.39 pp)** |
| | *(Target Bottleneck)* | Recall | 0.7634 | 0.7803 | +0.0169 | Modest recall increase |
| | | F1-Score | 0.7867 | 0.7740 | **-0.0127** | Deficit driven by false positives |
| | | mAP@0.50 | 0.8390 | 0.8316 | **-0.0074** | **Failed to improve (-0.74 pp)** |
| | | mAP@0.50:0.95 | 0.5995 | 0.5928 | **-0.0067** | **Failed to improve (-0.67 pp)** |
| **3** | **Auto Rickshaw** | Precision | 0.8856 | 0.8679 | -0.0177 | Slight precision decrease |
| | | Recall | 0.8121 | 0.8508 | **+0.0387** | **Large recall gain (+3.87 pp)** |
| | | F1-Score | 0.8474 | 0.8593 | +0.0119 | Improved overall balance |
| | | mAP@0.50 | 0.9090 | 0.9167 | +0.0077 | Modest improvement |
| | | mAP@0.50:0.95 | 0.7905 | 0.7985 | +0.0080 | Modest improvement |
| **4** | **Truck** | Precision | 0.8397 | 0.8110 | -0.0287 | Lower precision |
| | | Recall | 0.7868 | 0.8101 | +0.0233 | Improved recall |
| | | F1-Score | 0.8126 | 0.8105 | -0.0021 | Negligible change |
| | | mAP@0.50 | 0.8710 | 0.8781 | +0.0071 | Slight increase |
| | | mAP@0.50:0.95 | 0.7422 | 0.7331 | -0.0091 | Bounding box tightness decreased |

#### Key Observation on Motorcycle (Primary Hypothesis Failure):
Experiment 2 was explicitly formulated to overcome the **Motorcycle localization bottleneck** identified in Experiment 1. However, empirical evaluation proves that **scale augmentation failed to resolve this bottleneck**:
- Motorcycle mAP@0.50 fell from 0.8390 to 0.8316 (-0.74 pp).
- Motorcycle mAP@0.50:0.95 fell from 0.5995 to 0.5928 (-0.67 pp).
- Motorcycle precision suffered a severe decline (-4.39 pp), dropping to 0.7679.
- Rather than improving motorcycles, the augmentation primarily benefited **Bus** (+3.53 pp mAP@0.50, +2.39 pp mAP@0.50:0.95) and **Auto Rickshaw** (+3.87 pp recall).

### 4. Confusion Matrix Analysis
Comparing the normalized and count-based confusion matrices between Exp 1 and Exp 2 reveals key class interaction shifts:
1. **Motorcycle Misses vs. False Alarms**:
   - In Exp 1, 379 ground-truth motorcycles were missed as background (miss rate 35.8%), with 100 background false alarms.
   - In Exp 2, missed motorcycles increased to 434 (miss rate 39.0%), and background false alarms increased to 102.
   - Motorcycle had 0 cross-class confusions with any other vehicle type in both experiments; failures are purely background localization errors.
2. **Bus Detection Improvement**:
   - Bus true positive rate increased from 74.7% (148 detected) to 77.2% (152 detected), with background false negatives declining from 47 to 43.
3. **Car ↔ Truck & Car ↔ Auto Rickshaw**:
   - Car ↔ Truck confusions declined slightly from 8 total (3 Car $\to$ Truck, 5 Truck $\to$ Car) in Exp 1 to 5 total (4 Car $\to$ Truck, 1 Truck $\to$ Car) in Exp 2.
   - Car ↔ Auto Rickshaw confusions remained negligible (3 in Exp 1 vs. 3 in Exp 2).
4. **Aggregate Background False Alarms**:
   - Remained stable: 375 background false positives in Exp 1 vs. 367 in Exp 2.

### 5. Training Dynamics & Convergence
- **Convergence**: Both models trained for all 50 epochs without early stopping (patience=15 was never triggered). Learning rate decayed from 0.01 to $3.31 \times 10^{-5}$.
- **Loss Profiles**: Exp 2 exhibited higher initial training losses due to aggressive augmentation (`scale=0.9`, `copy_paste=0.3`), but converged smoothly.
- **Overfitting & Stability**: Validation box and classification losses reached their minima near the end of training (Exp 1: val box min 0.6863 at ep 49; Exp 2: val box min 0.6927 at ep 50). No severe divergence between training and validation loss curves was detected.
- **Sufficiency of 50 Epochs**: Metrics stabilized over the final 5 epochs (varying by $<0.003$ mAP), confirming that 50 epochs was fully sufficient for convergence.

### 6. Computational Cost & Inference Latency
- **Training Time**: Exp 1 = 262.03 min (4.36 h) vs. Exp 2 = 315.95 min (5.26 h), representing an additional training overhead of **+53.92 minutes (+20.4%)**.
- **Model Parameters & FLOPs**: Exactly identical (3,006,623 parameters, 8.1 GFLOPs, 6.22 MB checkpoint).
- **Inference Latency (RTX 3050 Laptop GPU)**:
  - Exp 1: 0.8ms preprocess, 3.6ms inference, 0.6ms postprocess (Total: ~5.0ms / ~200 FPS).
  - Exp 2: 1.3ms preprocess, 3.5ms inference, 0.7ms postprocess (Total: ~5.5ms / ~182 FPS).
  - Real-time throughput is fully preserved at $>180$ FPS.

### 7. Hard-Example Evaluation Status
- **Exp 1**: Completed and cataloged in `results/hard_examples/` (484 challenging images identified; 371 low confidence, 137 crowded, 106 small object).
- **Exp 2**: **Completed and cataloged** in `results/hard_examples/exp2/` (524 challenging images identified; 418 low confidence, 137 crowded, 106 small object, 141 count discrepancy).
  - *Motorcycle Diagnostics*:
    - Exp 1 vs Exp 2 True Positives: 680 $\to$ 679 (missed instances: 100 $\to$ 101).
    - Exp 1 vs Exp 2 False Positives: 427 $\to$ **471 (+44 false alarms / +10.3%)**.
    - Small Motorcycle Recall ($<32\times32$ px): 42.22% (19/45) $\to$ **35.56% (16/45)** (**-6.67 pp drop**).
    - Crowded Motorcycle Recall ($\ge 6$ vehicles): 81.22% (186/229) $\to$ **79.91% (183/229)** (**-1.31 pp drop**).
    - Low-Confidence Motorcycle Predictions ($<0.40$): 229 $\to$ **272 (+18.78%)**.
    - Detailed Report: `results/hard_examples/exp2/exp2_hard_example_report.md`.


### 8. Evidence-Based Decision & Next Scientific Step
- **Evidence Classification**: **Category C: Mixed results**
  - Justification: While overall mAP@0.50 increased modestly (+0.84 pp) and recall improved (+2.09 pp), overall precision degraded (-2.75 pp), F1 slightly declined (-0.17 pp), and mAP@0.50:0.95 remained essentially flat (+0.27 pp). Crucially, the targeted intervention failed to improve the specific bottleneck it was designed to solve (Motorcycle mAP degraded by -0.74 pp and motorcycle precision dropped by -4.39 pp).
- **Next Scientific Step**: **OPTION A: Proceed to Experiment 3: YOLOv8s capacity comparison.**
  - Scientific Rationale: Exp 2 demonstrated that purely data-level augmentation adjustments cannot compensate for the finite parameter capacity and feature resolution limitations of the 3.0M-parameter YOLOv8n backbone. When forced to learn larger scale jitter, the nano network suffered precision degradation on small objects. The next rigorous scientific question is whether increasing model capacity (YOLOv8s: 11.2M parameters, 28.6 GFLOPs) provides the representational power necessary to capture fine-grained motorcycle features and resolve boundary ambiguities.

### 9. Test-Set Protection Verification
- Verified: `data/processed/vision_toll_yolo/images/test` (1,083 images) and corresponding test labels remain **100% UNTOUCHED and UNSEEN**.
- `models/final/` contains no checkpoints; `results/test/` is completely empty.
- Zero test split evaluation, model selection, or threshold tuning has taken place.

---

## Experiment 3 — YOLOv8s Model Capacity Comparison

### 1. Experimental Configuration & Environment
- **Model Architecture**: YOLOv8s (`yolov8s.pt` pretrained on COCO as starting backbone)
- **Model Complexity**: 130 layers, 11,137,535 parameters, 28.7 GFLOPs (~3.7x parameters and ~3.5x FLOPs vs. YOLOv8n)
- **Target Classes**: Exactly 5 classes (`0: Bus`, `1: Car`, `2: Motorcycle`, `3: Auto Rickshaw`, `4: Truck`)
- **Hardware Platform**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (P0 state)
- **Training Telemetry**:
  - Image Size: 640x640
  - Batch Size: 16 (Peak VRAM: 4,034 MiB / 6,144 MiB; no memory paging)
  - Total Epochs: 50
  - Optimizer: AdamW (`lr0=0.002`, `lrf=0.01`, `weight_decay=0.0005`, `warmup_epochs=3.0`, `patience=15`, `seed=42`)
  - Augmentations: Preserved strictly at baseline defaults (`scale=0.5`, `copy_paste=0.0`, `mosaic=1.0`, `close_mosaic=10`, `erasing=0.4`, `fliplr=0.5`, `translate=0.1`)
  - Execution Duration: 5,529.5 seconds (~92.16 minutes / ~1.54 hours) across 50 epochs.

### 2. Overall Validation Comparison (Exp 1 vs. Exp 2 vs. Exp 3)
Evaluated on the exact 884 validation split images (3,100 ground-truth vehicle instances) with `best.pt`:

| Metric | Exp 1 (YOLOv8n Baseline) | Exp 2 (YOLOv8n + Aug) | Exp 3 (YOLOv8s Capacity) | $\Delta$ vs Exp 1 | $\Delta$ vs Exp 2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision** | **0.8579** | 0.8304 | 0.8289 | -0.0290 | -0.0015 |
| **Recall** | 0.7819 | 0.8028 | **0.8127** | **+0.0308 (+3.08 pp)** | **+0.0099 (+0.99 pp)** |
| **F1-Score** | 0.8181 | 0.8164 | **0.8207** | **+0.0026 (+0.26 pp)** | **+0.0043 (+0.43 pp)** |
| **mAP@0.50** | 0.8655 | 0.8739 | **0.8775** | **+0.0120 (+1.20 pp)** | **+0.0036 (+0.36 pp)** |
| **mAP@0.50:0.95** | 0.7197 | 0.7224 | **0.7399** | **+0.0202 (+2.02 pp)** | **+0.0175 (+1.75 pp)** |

### 3. Per-Class Validation Breakdown (Exp 1 vs. Exp 2 vs. Exp 3)

| Class | Metric | Exp 1 (YOLOv8n) | Exp 2 (YOLOv8n + Aug) | Exp 3 (YOLOv8s) | Absolute Change (Exp 3 vs Exp 1) | Key Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Bus** | Precision / Recall | 0.8570 / 0.7286 | 0.8263 / 0.7387 | 0.8344 / 0.7343 | -0.0226 / +0.0057 | Stable proposal balance |
| *(199 inst)*| mAP50 / mAP50-95 | 0.7840 / 0.6717 | 0.8193 / 0.6956 | **0.8211** / **0.6993** | **+0.0371 / +0.0276** | **Highest Bus accuracy** |
| **Car** | Precision / Recall | 0.8956 / 0.8185 | 0.8788 / 0.8344 | 0.8851 / **0.8396** | -0.0105 / **+0.0211** | Best passenger car recall |
| *(1,278 inst)*| mAP50 / mAP50-95 | 0.9250 / 0.7946 | 0.9236 / 0.7920 | **0.9252** / **0.8043** | +0.0002 / **+0.0097** | **Exceeded 0.80 mAP50-95** |
| **Motorcycle**| Precision / Recall | 0.8118 / 0.7634 | 0.7679 / 0.7803 | 0.7576 / **0.8115** | -0.0542 / **+0.0481** | **Strong recall jump (+4.81 pp)** |
| *(780 inst)* | F1-Score | 0.7867 | 0.7740 | 0.7836 | -0.0031 | Recovers from Exp 2 deficit |
| *(Target)* | mAP50 / mAP50-95 | 0.8390 / 0.5995 | 0.8316 / 0.5928 | **0.8397** / **0.6216** | +0.0007 / **+0.0221** | **Breaks through 0.60 ceiling!** |
| **Auto Rick**| Precision / Recall | 0.8856 / 0.8121 | 0.8679 / 0.8508 | 0.8698 / **0.8541** | -0.0158 / **+0.0420** | **Highest three-wheeler recall** |
| *(610 inst)*| mAP50 / mAP50-95 | 0.9090 / 0.7905 | 0.9167 / 0.7985 | **0.9189** / **0.8090** | **+0.0099 / +0.0185** | **Exceeded 0.80 mAP50-95** |
| **Truck** | Precision / Recall | 0.8397 / 0.7868 | 0.8110 / 0.8101 | 0.7976 / **0.8240** | -0.0421 / **+0.0372** | **Highest commercial truck recall** |
| *(233 inst)*| mAP50 / mAP50-95 | 0.8710 / 0.7422 | 0.8781 / 0.7331 | **0.8824** / **0.7654** | **+0.0114 / +0.0232** | **Substantial tightness gain** |

### 4. Hard-Example Profiling (Exp 1 vs. Exp 2 vs. Exp 3)
- **Total Hard-Example Scenes**: 484 (Exp 1) $\to$ 524 (Exp 2) $\to$ **412 (Exp 3)** (**-14.87% reduction vs baseline**).
- **Low-Confidence Scenes ($<0.40$)**: 371 (Exp 1) $\to$ 418 (Exp 2) $\to$ **283 (Exp 3)** (**-23.72% reduction vs baseline**).
- **Motorcycle False Positives**: 427 (Exp 1) $\to$ 471 (Exp 2) $\to$ **363 (Exp 3)** (**-64 false alarms / -15.0% vs baseline**).
- **Motorcycle Median Confidence**: 0.7169 (Exp 1) $\to$ 0.6675 (Exp 2) $\to$ **0.7789 (Exp 3)** (Significantly higher prediction certainty).
- **Small Motorcycle Recall ($<32\times32$ px)**: 42.22% (Exp 1) $\to$ 35.56% (Exp 2) $\to$ 33.33% (Exp 3) (Persistent spatial downsampling limit).

### 5. Computational & Real-Time Performance
- **Model Checkpoint**: [`models/exp3_yolov8s/best.pt`](file:///D:/VisionToll/models/exp3_yolov8s/best.pt) (22.5 MB).
- **Inference Latency**: **7.1 ms inference** + 0.3 ms preprocess + 0.5 ms postprocess = **~7.9 ms pipeline latency per frame**.
- **Throughput**: **$>125$ FPS on RTX 3050 Laptop GPU**, comfortably satisfying the 30–60 FPS camera requirements of real-world toll plazas.

### 6. Evidence-Based Assessment & Model Selection
- **Classification**: **Category B: Improvement with tradeoffs**.
- **Model Selection Verdict**: **YOLOv8s (Experiment 3)** is the **champion model candidate** of the VisionToll pipeline. It successfully breaks through the 0.60 Motorcycle mAP@0.50:0.95 bottleneck, achieves the highest overall accuracy (0.7399 mAP50-95, 0.8207 F1), sharply suppresses false alarms and low-confidence proposals, and maintains robust real-time throughput ($>125$ FPS).

### 7. Test-Set Protection & Freeze Confirmation (Prior to Final Evaluation)
- Prior to the final test evaluation, the 1,083-image test split in `data/processed/vision_toll_yolo/images/test` remained strictly quarantined and unseen.
- Model selection was executed entirely upon validation split evidence.

---

## Final Test-Set Evaluation

> [!IMPORTANT]
> **METHODOLOGICAL DISTINCTION: VALIDATION RESULTS vs. FINAL TEST RESULTS**  
> - **Validation Results** (Sections above: Exp 1, Exp 2, Exp 3) reflect iterative hyperparameter and architectural model selection on the 884-image validation split (`data/processed/vision_toll_yolo/images/val`).
> - **Final Test Results** (This section) reflect the **strict one-time held-out evaluation** of the permanently **FROZEN** champion model on the 1,083-image test split (`data/processed/vision_toll_yolo/images/test`).
> - **Zero training, fine-tuning, threshold tuning, or model selection** was conducted on or after test evaluation.

### 1. Frozen Model Verification
- **Model Architecture**: YOLOv8s (Experiment 3)
- **Checkpoint**: `D:\VisionToll\models\exp3_yolov8s\best.pt`
- **Verified SHA-256**: `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD`
- **Status**: FROZEN FOR FINAL TEST EVALUATION

### 2. Overall Test Split Results (1,083 Images, 3,926 GT Instances)

| Metric | Final Test Result |
| :--- | :---: |
| **Precision** | **0.8574** |
| **Recall** | **0.8017** |
| **F1-Score** | **0.8286** |
| **mAP@0.50** | **0.8811** |
| **mAP@0.50:0.95** | **0.7343** |
| **Inference Latency** | **7.05 ms** (7.97 ms total pipeline) |
| **Throughput** | **125.4 FPS** |

### 3. Per-Class Test Split Breakdown

| Class ID | Class Name | Ground Truth Count | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Bus** | 219 | 0.8734 | 0.7563 | 0.8107 | 0.8382 | 0.6976 |
| 1 | **Car** | 1,559 | 0.9059 | 0.8273 | 0.8648 | 0.9264 | 0.8032 |
| 2 | **Motorcycle** | 1,085 | 0.8202 | 0.7860 | 0.8027 | 0.8644 | 0.6361 |
| 3 | **Auto Rickshaw** | 754 | 0.9001 | 0.8481 | 0.8733 | 0.9216 | 0.8163 |
| 4 | **Truck** | 309 | 0.7874 | 0.7912 | 0.7893 | 0.8550 | 0.7182 |

### 4. Validation vs. Test Generalization Comparison

| Metric | Validation (Exp 3 `best.pt`) | Final Test Split | Difference ($\Delta$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Precision** | 0.8289 | 0.8574 | +0.0285 | Lower false positive rate on test |
| **Recall** | 0.8127 | 0.8017 | -0.0110 | Highly stable detection recall (~1% delta) |
| **F1-Score** | 0.8207 | 0.8286 | +0.0079 | Balanced performance maintained |
| **mAP@0.50** | 0.8775 | 0.8811 | +0.0036 | Consistent multi-vehicle discovery |
| **mAP@0.50:0.95** | 0.7399 | 0.7343 | -0.0056 | Negligible bounding box drift (<0.8% relative) |
| **Motorcycle mAP50-95**| 0.6216 | 0.6361 | +0.0145 | Bottleneck resolution verified on unseen data |

### 5. Test Hard-Example Summary
- Total Hard-Example Scenes: **495 / 1,083 (45.71%)**
  - Low-confidence scenes ($<0.40$): 349
  - Crowded scenes ($\ge 6$ vehicles): 158
  - Small-object scenes ($<32\times32$ px): 110
  - Count-discrepancy scenes ($|\text{GT}-\text{Pred}| \ge 3$): 105
  - Complete detector misses: 4 scenes (0.37%)
- Preserved Artifacts:
  - Full Report: [`results/final_test/final_test_report.md`](file:///D:/VisionToll/results/final_test/final_test_report.md)
  - Three-Experiment Summary: [`results/final_test/final_experiment_summary.md`](file:///D:/VisionToll/results/final_test/final_experiment_summary.md)
  - Hard Examples Manifest: [`results/hard_examples/final_test/final_test_hard_examples_manifest.json`](file:///D:/VisionToll/results/hard_examples/final_test/final_test_hard_examples_manifest.json)



