# VisionToll: Research Methodology & Experimental Design

## 1. Research Motivation & Problem Statement
Automated toll plaza monitoring requires rapid, real-time visual vehicle detection and multi-class categorization. In developing nations like India, toll plazas encounter high vehicular density, chaotic lane discipline, and extreme fine-grained diversity across five primary operational categories: **Bus, Car, Motorcycle, Auto Rickshaw, and Truck**.

Traditional approaches rely on intrusive physical axle sensors or loop detectors that suffer from mechanical wear and lack semantic awareness. VisionToll demonstrates a deep learning camera-based pipeline that processes vehicle images and performs robust simultaneous localization and classification.

---

## 2. Scientific Workflow & Experimental Stages

```
Raw FGVD Dataset (Zenodo 7488960)
         │
         ▼
Five-Class Ground Truth Extraction (Pascal-VOC XML)
         │
         ▼
YOLOv8 Dataset Conversion & Coordinate Normalization
         │
         ▼
Empirical Dataset Audit (Geometry, Balance, Quality)
         │
         ▼
Visual Ground-Truth Verification (Contact Sheets)
         │
         ▼
Experiment 1: YOLOv8n Baseline Training (Local RTX 3050 GPU)
         │
         ▼
Validation Evaluation & Hard-Example / Error Analysis
         │
         ▼
Experiment 2: YOLOv8n Targeted Improvement
         │
         ▼
Experiment 3: YOLOv8s Model Capacity Comparison
         │
         ▼
Comparative Validation Analysis & Final Model Selection
         │
         ▼
Untouched Final Test Set Evaluation
         │
         ▼
Final Error Taxonomy & Discussion
         │
         ▼
Image-Based Streamlit Application
```

---

## 3. Class Definitions & Target Taxonomy
VisionToll enforces exactly five target vehicle classes:
- **Class 0: Bus** — Public and private passenger transit buses and heavy transit coaches. (Mini-buses are strictly excluded to preserve transit bus feature integrity).
- **Class 1: Car** — Sedans, hatchbacks, SUVs, MUVs, private passenger cars.
- **Class 2: Motorcycle** — Two-wheeled motorized motorcycles with step-over frames. (Scooters are strictly excluded due to distinct step-through morphology and are never remapped).
- **Class 3: Auto Rickshaw** — Motorized three-wheeled commercial passenger autorickshaws.
- **Class 4: Truck** — Heavy commercial freight vehicles, rigid trucks, multi-axle lorries, haulers.

---

## 4. Controlled Experimental Protocol
All training experiments use identical random seed (42), resolution (640x640), batch size (16), optimizer (SGD/auto), and split assignments.
- **Experiment 1 (Baseline)**: YOLOv8n default fine-tuning for 50 epochs (imgsz 640, batch 16, seed 42, AdamW optimizer) executed and validated exclusively on the 884-image validation set. Completed with mAP@0.50 = 0.8655 and mAP@0.50:0.95 = 0.7197. Identified Motorcycle fine-localization and Bus occlusion as primary failure modes.
- **Experiment 2 (Targeted Improvement)**: YOLOv8n with scale-augmentation intervention (`scale=0.9`, `copy_paste=0.3`) targeting small vehicle resolution and motorcycle bottlenecks. Completed with mAP@0.50 = 0.8739 and mAP@0.50:0.95 = 0.7224. Resulted in mixed outcomes: significant gains in Bus (+3.53 pp mAP50) and Auto Rickshaw (+3.87 pp recall), but worsened Motorcycle mAP (-0.74 pp) and precision (-4.39 pp), highlighting representational capacity bottlenecks in the 3.0M-parameter nano model.
- **Model Selection & Final Test Evaluation**: Comparative validation evidence conclusively established YOLOv8s (Experiment 3) as the champion detector. The model checkpoint was cryptographically verified (SHA-256: `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD`) and permanently frozen. The held-out test split (1,083 images, 3,926 GT instances) was evaluated strictly once, yielding **0.8811 mAP@0.50**, **0.7343 mAP@0.50:0.95**, **0.8286 F1**, and **125.4 FPS**, confirming excellent generalization without post-test modifications.



