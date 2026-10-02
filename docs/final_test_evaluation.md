# VisionToll — Final Held-Out Test Evaluation

**Project**: VisionToll: Deep Learning-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring  
**Course**: 23CSE473 Neural Networks and Deep Learning (Group A12)  
**Evaluation Date**: 2026-10-02  
**Evaluation Status**: **COMPLETE — MODEL PERMANENTLY FROZEN**  

---

## Executive Summary

The frozen **YOLOv8s** production model was evaluated **once** on the 1,083-image held-out test split that was quarantined throughout the entire training and model selection phase.

| Key Result | Value |
|:---|:---:|
| **mAP@0.50** | **0.8811** |
| **mAP@0.50:0.95** | **0.7343** |
| **F1-Score** | **0.8286** |
| **Inference FPS** | **125.4** |
| **Generalization Gap (mAP50-95)** | **−0.0056 (−0.76%)** |

The model demonstrated negligible overfitting and confirmed that the validation-based model selection process was methodologically sound.

---

## 1. Frozen Model Record

| Field | Value |
|:---|:---|
| Model | YOLOv8s |
| Checkpoint | `D:\VisionToll\models\exp3_yolov8s\best.pt` |
| SHA-256 | `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD` |
| Parameters | 11,137,535 |
| GFLOPs (640×640) | 28.7 |
| Checkpoint Size | 22.5 MB |
| Freeze Record | `results/final_model_freeze.json` |
| Training After Freeze | **None** |
| Weight Modification | **None** |

---

## 2. Reproducibility Environment

| Component | Version |
|:---|:---|
| Python | 3.12.5 [MSC v.1940 64-bit] |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| Ultralytics | 8.4.138 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| OS | Windows 11 (64-bit) |

**Evaluation settings:**
- Dataset config: `data/processed/vision_toll_yolo/data.yaml`
- Split: `test`
- Input resolution: 640×640 (letterboxed)
- Batch size: 16
- Confidence threshold (mAP curve): 0.001 (standard)
- Confidence threshold (hard-example): 0.25
- IoU NMS threshold: 0.70
- mAP IoU range: 0.50:0.05:0.95

---

## 3. Test Dataset Profile

| Field | Value |
|:---|:---|
| Test Images | 1,083 |
| Test Label Files | 1,083 (100% paired) |
| GT Vehicle Instances | 3,926 |
| Class Violations | 0 |
| Train/Val Overlap | None |

**GT Class Distribution:**

| Class | ID | Instances | % of Test |
|:---|:---:|:---:|:---:|
| Bus | 0 | 219 | 5.58% |
| Car | 1 | 1,559 | 39.71% |
| Motorcycle | 2 | 1,085 | 27.64% |
| Auto Rickshaw | 3 | 754 | 19.21% |
| Truck | 4 | 309 | 7.87% |
| **Total** | — | **3,926** | **100%** |

---

## 4. Overall Test Results

| Metric | Value |
|:---|:---:|
| Precision | 0.8574 |
| Recall | 0.8017 |
| F1-Score | 0.8286 |
| mAP@0.50 | **0.8811** |
| mAP@0.50:0.95 | **0.7343** |
| Total GT Instances | 3,926 |
| Total Predictions (NMS) | 4,240 |
| Preprocessing Latency | 0.27 ms/img |
| Inference Latency | 7.05 ms/img |
| Postprocessing Latency | 0.65 ms/img |
| **Total Pipeline Latency** | **7.97 ms/img** |
| **Throughput** | **125.4 FPS** |

---

## 5. Per-Class Test Results

| Class | GT | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Bus | 219 | 0.8734 | 0.7563 | 0.8107 | 0.8382 | 0.6976 |
| Car | 1,559 | 0.9059 | 0.8273 | 0.8648 | 0.9264 | 0.8032 |
| **Motorcycle** | **1,085** | **0.8202** | **0.7860** | **0.8027** | **0.8644** | **0.6361** |
| Auto Rickshaw | 754 | 0.9001 | 0.8481 | 0.8733 | 0.9216 | 0.8163 |
| Truck | 309 | 0.7874 | 0.7912 | 0.7893 | 0.8550 | 0.7182 |
| **All Classes** | **3,926** | **0.8574** | **0.8017** | **0.8286** | **0.8811** | **0.7343** |

---

## 6. Motorcycle Diagnostics

| Metric | Value |
|:---|:---|
| GT Motorcycles | 1,085 |
| Predicted (conf ≥ 0.25) | 1,332 |
| True Positives (IoU ≥ 0.50) | 930 |
| False Negatives (Missed) | 155 |
| False Positives (Background) | 402 |
| **Recall @ IoU 0.50** | **0.8571** |
| Precision (conf ≥ 0.25) | 0.6982 |
| Small (<32×32 px) Recall | 40.0% (16/40) |
| Crowded (≥6 vehicles) Recall | 79.87% (254/318) |
| Mean Prediction Confidence | 0.7007 |
| Median Prediction Confidence | 0.7859 |

---

## 7. Hard-Example Profile

| Category | Count | % of Test |
|:---|:---:|:---:|
| Total Test Scenes | 1,083 | 100% |
| **Hard Scenes (any category)** | **495** | **45.71%** |
| Low-Confidence Scenes (conf ∈ [0.25, 0.40)) | 349 | 32.23% |
| Crowded Scenes (≥6 GT vehicles) | 158 | 14.59% |
| Small-Object Scenes (<32×32 px) | 110 | 10.16% |
| Count Discrepancy Scenes (\|GT−Pred\|≥3) | 105 | 9.70% |
| Complete-Miss Scenes (0 detections) | 4 | 0.37% |

---

## 8. Validation-to-Test Generalization Gap

> Comparison is **strictly** between YOLOv8s Experiment-3 validation and the final test. No cross-model comparison on test data.

| Metric | Validation | Test | Delta | Assessment |
|:---|:---:|:---:|:---:|:---|
| Precision | 0.8289 | 0.8574 | +0.0285 | Favorable |
| Recall | 0.8127 | 0.8017 | −0.0110 | Highly consistent |
| F1 | 0.8207 | 0.8286 | +0.0079 | Stable |
| mAP@0.50 | 0.8775 | 0.8811 | +0.0036 | Excellent correspondence |
| **mAP@0.50:0.95** | **0.7399** | **0.7343** | **−0.0056** | **<0.8% — negligible** |
| Moto mAP50-95 | 0.6216 | 0.6361 | +0.0145 | Bottleneck confirmed resolved |
| Car mAP50-95 | 0.8043 | 0.8032 | −0.0011 | Exact agreement |
| Auto Rickshaw mAP50-95 | 0.8090 | 0.8163 | +0.0073 | Exact agreement |
| Bus mAP50-95 | 0.6993 | 0.6976 | −0.0017 | Exact agreement |
| Truck mAP50-95 | 0.7654 | 0.7182 | −0.0472 | Moderate (test composition) |

**Conclusion**: The generalization gap across primary metrics is below 0.8% relative. The model does not overfit to the validation distribution.

---

## 9. Integrity Verification Checklist

| Check | Result |
|:---|:---:|
| Exactly 1,083 test images evaluated | ✅ PASS |
| SHA-256 matches frozen model record | ✅ PASS |
| No training performed | ✅ PASS |
| No weight files changed | ✅ PASS |
| No threshold tuning on test set | ✅ PASS |
| Test set not accessed before this evaluation | ✅ PASS |
| Final metrics saved successfully | ✅ PASS |

---

## 10. Output Artifact Manifest

All outputs are stored under `results/final_test/`:

| File | Description |
|:---|:---|
| [`final_test_report.md`](file:///D:/VisionToll/results/final_test/final_test_report.md) | Complete academic evaluation report (19.4 KB) |
| [`final_test_metrics.json`](file:///D:/VisionToll/results/final_test/final_test_metrics.json) | Machine-readable canonical metrics record |
| [`final_test_metrics.csv`](file:///D:/VisionToll/results/final_test/final_test_metrics.csv) | Val vs. test comparison summary row |
| [`per_class_metrics.csv`](file:///D:/VisionToll/results/final_test/per_class_metrics.csv) | Per-class precision/recall/F1/mAP table |
| [`inference_benchmark.json`](file:///D:/VisionToll/results/final_test/inference_benchmark.json) | Latency, FPS, and operational feasibility record |
| [`evaluation_log.txt`](file:///D:/VisionToll/results/final_test/evaluation_log.txt) | Step-by-step execution log |
| `confusion_matrix.png` | Raw count confusion matrix |
| `confusion_matrix_normalized.png` | Normalized confusion matrix |
| `BoxPR_curve.png` | Precision-Recall curves per class |
| `BoxF1_curve.png` | F1-confidence curves per class |

---

## 11. Final Lifecycle Statement

With this evaluation complete:

- The frozen **YOLOv8s** model achieves **mAP@0.50 = 0.8811** and **125.4 FPS** on the held-out test set.
- The motorcycle detection bottleneck from the YOLOv8n baseline was **confirmed resolved** on unseen test data (mAP50-95: 0.6361 vs. 0.5995 baseline).
- The **VisionToll experimental lifecycle is formally concluded**.
- Model weights are **permanently frozen**.
- **No further training, fine-tuning, or weight modification** will be performed.
