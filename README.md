# VisionToll
Deep Learning-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring

## 1. Project Overview
VisionToll is an intelligent, image-based vehicle detection and classification system designed to streamline toll plaza operations. It uses deep learning to identify and categorize vehicles in highway scenes.

## 2. Problem Statement
Manual toll collection and auditing are slow and prone to errors. Automated systems need accurate classification of diverse vehicle types under varying conditions.

## 3. Objectives
To build a robust model capable of detecting and classifying key vehicle types with high precision and real-time inference speed.

## 4. Dataset
- **Dataset used**: IIIT-H Fine-Grained Vehicle Detection (FGVD)
- **Size**: 5,502 images
- *Note: The full dataset is intentionally omitted from this GitHub repository due to its large size. The full dataset remains in the original project archive.*

## 5. Vehicle Classes
The model detects 5 target classes:
- Bus
- Car
- Motorcycle
- Auto Rickshaw
- Truck

## 6. Methodology
We fine-tuned multiple state-of-the-art object detection models using transfer learning, assessing them based on localization tightness, classification accuracy, and real-time inference feasibility.

## 7. Six Model Experiments
We experimented with the following architectures:
- ARCH-01 YOLOv8n
- ARCH-02 YOLOv8s
- ARCH-03 YOLOv5s
- ARCH-04 Faster R-CNN ResNet50-FPN
- ARCH-05 SSD-Lite MobileNetV3 Large
- ARCH-06 RetinaNet ResNet50-FPN

YOLOv8s provided the best balance of mAP and FPS. *(See `results/final_model_comparison.md` for full metrics).*

## 8. Final Model
**YOLOv8s** is the final production model. The frozen checkpoint is located at `models/exp3_yolov8s/best.pt`.

## 9. Final Test Results
Based on our held-out test split, YOLOv8s achieved:
- mAP@0.50: 88.11%
- mAP@0.50:0.95: 73.43%
- Precision: 85.74%
- Recall: 80.17%

## 10. Streamlit Application
A simple web application is provided for demonstration. It loads the frozen model and performs inference on static images.

## 11. Installation
```bash
pip install -r requirements.txt
```

## 12. How to Run
```bash
streamlit run app.py
```

## 13. Project Structure
- `app.py`: Streamlit application
- `models/`: Contains the final YOLOv8s checkpoint
- `results/`: Contains evaluation metrics, final comparisons, and visualizations
- `scripts/`, `src/`: Source code and tools
- `docs/`: Additional documentation

## 14. Visualizations
Performance graphs and comparisons are available in `results/visualizations/`.

## 15. Limitations
- Difficulty resolving highly overlapping vehicles in extremely dense traffic.
- Small/distant vehicles may be missed.

## 16. Future Work
- Incorporating video tracking (e.g., DeepSORT) for counting vehicles across frames.
- Expanding the taxonomy.

## 17. Dataset Setup
As the dataset is not bundled, users wishing to retrain must manually acquire the FGVD dataset. This repository is not fully self-contained for retraining out-of-the-box.

## 18. Model Information
- Size: ~22MB
- Parameters: ~11.1M
- Output: 5 classes + bounding boxes
