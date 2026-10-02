# VisionToll: Top 5 Validation Demo Candidates & Model Verification

**Generated**: Autonomous Analysis Pipeline  
**Source Split**: Validation Set (`data/processed/vision_toll_yolo/images/val`)  
**Test Set Isolation**: STRICTLY UNTOUCHED (Zero test images used or evaluated)  
**Frozen Model**: `models/exp3_yolov8s/best.pt`  
**Verified SHA-256**: `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD`  
**Inference Settings**: Resolution 640×640, Confidence Threshold = 0.25, NMS IoU = 0.50  

---

## 1. Executive Summary of Selected Demo Candidates

| Candidate | Original File | Resolution | GT Count | GT Classes | Pred Count | Pred Classes | Highlight |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **demo_01.jpg** | `733.jpg` | 1920x1080 | 9 | Bus, Car, Motorcycle, Auto Rickshaw, Truck | 7 | Bus, Car, Motorcycle, Auto Rickshaw, Truck | 5-Class Complete Census Scene |
| **demo_02.jpg** | `291.jpg` | 1280x720 | 6 | Bus, Car, Motorcycle, Auto Rickshaw, Truck | 5 | Bus, Car, Motorcycle, Auto Rickshaw, Truck | 5-Class Toll Plaza Lane View |
| **demo_03.jpg** | `4273.jpg` | 1920x1080 | 8 | Bus, Car, Motorcycle, Auto Rickshaw | 9 | Bus, Car, Motorcycle, Auto Rickshaw | High-Density Motorcycle & Transit Scene |
| **demo_04.jpg** | `2371.jpg` | 1920x1080 | 4 | Bus, Car, Motorcycle, Auto Rickshaw | 4 | Bus, Car, Motorcycle, Auto Rickshaw | Clean 4-Class Exhibition Scene |
| **demo_05.jpg** | `1181.jpg` | 1920x1080 | 8 | Bus, Car, Motorcycle, Auto Rickshaw | 9 | Bus, Car, Motorcycle, Auto Rickshaw | Dense Urban Multiclass Queue |

---

## 2. Detailed Candidate Profiles & Verification

### Candidate 01: `demo_01.jpg` (Source: `733.jpg`)
**Profile Description**: 5-Class Complete Census Scene (Full HD 1080p, all 5 target classes present and detected, prominent foreground bus and cars)  
**Image Dimensions**: 1920x1080 (Aspect Ratio: 1.78:1)  

#### Ground Truth Statistics:
- **Filename**: `demo_01.jpg` (Original: `733.jpg`)
- **Classes present**: Bus, Car, Motorcycle, Auto Rickshaw, Truck (5 distinct classes)
- **Vehicle count**: 9
- **Motorcycle count**: 1
- **Auto Rickshaw count**: 2
- **Bus count**: 2
- **Car count**: 3
- **Truck count**: 1
- **Largest box**: 824,356.6 px²
- **Smallest box**: 4,875.9 px²
- **Average box area**: 187,902.86666666667 px²
- **Why it is suitable for demonstration**: 5-Class Complete Census Scene (Full HD 1080p, all 5 target classes present and detected, prominent foreground bus and cars). It provides exceptional visual clarity, representative bounding box scales for live visual demonstration, and tests multiple distinct vehicle geometries under standard surveillance viewpoints.

#### Model Verification (Frozen YOLOv8s @ conf=0.25, iou=0.50):
- **Ground-truth classes**: Bus, Car, Motorcycle, Auto Rickshaw, Truck
- **Predicted classes**: Bus, Car, Motorcycle, Auto Rickshaw, Truck
- **Total predictions**: 7
- **Per-class predictions**: Bus: 1, Car: 2, Motorcycle: 1, Auto Rickshaw: 2, Truck: 1
- **Detections and Confidence Values**:
  1. **Bus** — Confidence: `0.9614` | Box `[x1=0.0, y1=3.8, x2=768.9, y2=1071.5]` | Size: `768.9×1067.7 px` (Area: 820,964.9 px²)
  2. **Car** — Confidence: `0.9331` | Box `[x1=860.1, y1=493.4, x2=1489.2, y2=1018.5]` | Size: `629.2×525.2 px` (Area: 330,427.2 px²)
  3. **Car** — Confidence: `0.9102` | Box `[x1=1504.3, y1=517.0, x2=1919.8, y2=1067.7]` | Size: `415.5×550.7 px` (Area: 228,819.0 px²)
  4. **Auto Rickshaw** — Confidence: `0.8971` | Box `[x1=1332.4, y1=475.0, x2=1701.9, y2=744.1]` | Size: `369.4×269.1 px` (Area: 99,431.8 px²)
  5. **Auto Rickshaw** — Confidence: `0.7060` | Box `[x1=1820.9, y1=456.7, x2=1918.8, y2=566.4]` | Size: `97.9×109.7 px` (Area: 10,737.0 px²)
  6. **Truck** — Confidence: `0.5272` | Box `[x1=813.4, y1=293.3, x2=1274.2, y2=707.8]` | Size: `460.8×414.6 px` (Area: 191,025.1 px²)
  7. **Motorcycle** — Confidence: `0.2540` | Box `[x1=1679.9, y1=518.7, x2=1769.4, y2=667.6]` | Size: `89.4×148.9 px` (Area: 13,321.4 px²)

---

### Candidate 02: `demo_02.jpg` (Source: `291.jpg`)
**Profile Description**: 5-Class Toll Plaza Lane View (720p, all 5 target classes present and detected, high-confidence truck and transit vehicles)  
**Image Dimensions**: 1280x720 (Aspect Ratio: 1.78:1)  

#### Ground Truth Statistics:
- **Filename**: `demo_02.jpg` (Original: `291.jpg`)
- **Classes present**: Bus, Car, Motorcycle, Auto Rickshaw, Truck (5 distinct classes)
- **Vehicle count**: 6
- **Motorcycle count**: 1
- **Auto Rickshaw count**: 1
- **Bus count**: 2
- **Car count**: 1
- **Truck count**: 1
- **Largest box**: 174,024.6 px²
- **Smallest box**: 2,295.0 px²
- **Average box area**: 54,209.4 px²
- **Why it is suitable for demonstration**: 5-Class Toll Plaza Lane View (720p, all 5 target classes present and detected, high-confidence truck and transit vehicles). It provides exceptional visual clarity, representative bounding box scales for live visual demonstration, and tests multiple distinct vehicle geometries under standard surveillance viewpoints.

#### Model Verification (Frozen YOLOv8s @ conf=0.25, iou=0.50):
- **Ground-truth classes**: Bus, Car, Motorcycle, Auto Rickshaw, Truck
- **Predicted classes**: Bus, Car, Motorcycle, Auto Rickshaw, Truck
- **Total predictions**: 5
- **Per-class predictions**: Bus: 1, Car: 1, Motorcycle: 1, Auto Rickshaw: 1, Truck: 1
- **Detections and Confidence Values**:
  1. **Bus** — Confidence: `0.9462` | Box `[x1=109.7, y1=157.4, x2=461.7, y2=526.2]` | Size: `352.0×368.8 px` (Area: 129,826.5 px²)
  2. **Truck** — Confidence: `0.9314` | Box `[x1=638.6, y1=256.1, x2=1192.2, y2=568.8]` | Size: `553.6×312.7 px` (Area: 173,108.7 px²)
  3. **Auto Rickshaw** — Confidence: `0.8945` | Box `[x1=0.0, y1=350.0, x2=86.7, y2=521.8]` | Size: `86.7×171.8 px` (Area: 14,894.5 px²)
  4. **Motorcycle** — Confidence: `0.6456` | Box `[x1=538.4, y1=359.2, x2=611.2, y2=408.9]` | Size: `72.9×49.7 px` (Area: 3,620.2 px²)
  5. **Car** — Confidence: `0.3004` | Box `[x1=513.2, y1=340.7, x2=556.7, y2=376.4]` | Size: `43.4×35.7 px` (Area: 1,550.9 px²)

---

### Candidate 03: `demo_03.jpg` (Source: `4273.jpg`)
**Profile Description**: High-Density Motorcycle & Transit Scene (Full HD 1080p, 4 target classes, 4 distinct high-confidence motorcycles detected alongside bus, auto rickshaw, and cars)  
**Image Dimensions**: 1920x1080 (Aspect Ratio: 1.78:1)  

#### Ground Truth Statistics:
- **Filename**: `demo_03.jpg` (Original: `4273.jpg`)
- **Classes present**: Bus, Car, Motorcycle, Auto Rickshaw (4 distinct classes)
- **Vehicle count**: 8
- **Motorcycle count**: 4
- **Auto Rickshaw count**: 1
- **Bus count**: 1
- **Car count**: 2
- **Truck count**: 0
- **Largest box**: 266,266.1 px²
- **Smallest box**: 13,529.0 px²
- **Average box area**: 139,042.375 px²
- **Why it is suitable for demonstration**: High-Density Motorcycle & Transit Scene (Full HD 1080p, 4 target classes, 4 distinct high-confidence motorcycles detected alongside bus, auto rickshaw, and cars). It provides exceptional visual clarity, representative bounding box scales for live visual demonstration, and tests multiple distinct vehicle geometries under standard surveillance viewpoints.

#### Model Verification (Frozen YOLOv8s @ conf=0.25, iou=0.50):
- **Ground-truth classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Predicted classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Total predictions**: 9
- **Per-class predictions**: Bus: 1, Car: 3, Motorcycle: 4, Auto Rickshaw: 1, Truck: 0
- **Detections and Confidence Values**:
  1. **Auto Rickshaw** — Confidence: `0.9487` | Box `[x1=718.4, y1=494.5, x2=1140.5, y2=1001.8]` | Size: `422.0×507.2 px` (Area: 214,077.1 px²)
  2. **Bus** — Confidence: `0.9434` | Box `[x1=0.0, y1=303.5, x2=651.8, y2=772.1]` | Size: `651.8×468.6 px` (Area: 305,432.2 px²)
  3. **Car** — Confidence: `0.9139` | Box `[x1=1.9, y1=587.4, x2=476.7, y2=923.4]` | Size: `474.8×336.0 px` (Area: 159,561.9 px²)
  4. **Motorcycle** — Confidence: `0.9067` | Box `[x1=325.4, y1=698.3, x2=626.0, y2=1080.0]` | Size: `300.6×381.7 px` (Area: 114,747.7 px²)
  5. **Motorcycle** — Confidence: `0.8987` | Box `[x1=1250.3, y1=671.8, x2=1464.2, y2=1034.6]` | Size: `213.8×362.7 px` (Area: 77,567.1 px²)
  6. **Car** — Confidence: `0.8872` | Box `[x1=1172.8, y1=581.2, x2=1563.6, y2=856.9]` | Size: `390.7×275.6 px` (Area: 107,693.7 px²)
  7. **Motorcycle** — Confidence: `0.8300` | Box `[x1=634.4, y1=648.3, x2=728.3, y2=822.7]` | Size: `94.0×174.3 px` (Area: 16,385.9 px²)
  8. **Motorcycle** — Confidence: `0.7589` | Box `[x1=1474.3, y1=742.4, x2=1918.9, y2=1079.7]` | Size: `444.6×337.3 px` (Area: 149,956.1 px²)
  9. **Car** — Confidence: `0.4987` | Box `[x1=0.0, y1=753.6, x2=192.6, y2=1080.0]` | Size: `192.6×326.4 px` (Area: 62,853.8 px²)

---

### Candidate 04: `demo_04.jpg` (Source: `2371.jpg`)
**Profile Description**: Clean 4-Class Exhibition Scene (Full HD 1080p, perfect 1:1 detection across Bus, Car, Motorcycle, and Auto Rickshaw with all confidences > 0.86)  
**Image Dimensions**: 1920x1080 (Aspect Ratio: 1.78:1)  

#### Ground Truth Statistics:
- **Filename**: `demo_04.jpg` (Original: `2371.jpg`)
- **Classes present**: Bus, Car, Motorcycle, Auto Rickshaw (4 distinct classes)
- **Vehicle count**: 4
- **Motorcycle count**: 1
- **Auto Rickshaw count**: 1
- **Bus count**: 1
- **Car count**: 1
- **Truck count**: 0
- **Largest box**: 540,939.6 px²
- **Smallest box**: 21,120.1 px²
- **Average box area**: 240,932.19999999998 px²
- **Why it is suitable for demonstration**: Clean 4-Class Exhibition Scene (Full HD 1080p, perfect 1:1 detection across Bus, Car, Motorcycle, and Auto Rickshaw with all confidences > 0.86). It provides exceptional visual clarity, representative bounding box scales for live visual demonstration, and tests multiple distinct vehicle geometries under standard surveillance viewpoints.

#### Model Verification (Frozen YOLOv8s @ conf=0.25, iou=0.50):
- **Ground-truth classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Predicted classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Total predictions**: 4
- **Per-class predictions**: Bus: 1, Car: 1, Motorcycle: 1, Auto Rickshaw: 1, Truck: 0
- **Detections and Confidence Values**:
  1. **Bus** — Confidence: `0.9645` | Box `[x1=560.7, y1=178.8, x2=1302.4, y2=918.7]` | Size: `741.7×739.9 px` (Area: 548,812.1 px²)
  2. **Car** — Confidence: `0.9345` | Box `[x1=1368.7, y1=520.4, x2=1919.9, y2=1078.8]` | Size: `551.2×558.4 px` (Area: 307,784.3 px²)
  3. **Motorcycle** — Confidence: `0.8897` | Box `[x1=0.2, y1=725.9, x2=277.8, y2=1077.7]` | Size: `277.6×351.8 px` (Area: 97,672.4 px²)
  4. **Auto Rickshaw** — Confidence: `0.8643` | Box `[x1=1362.9, y1=549.3, x2=1534.8, y2=709.3]` | Size: `171.9×160.0 px` (Area: 27,507.1 px²)

---

### Candidate 05: `demo_05.jpg` (Source: `1181.jpg`)
**Profile Description**: Dense Urban Multiclass Queue (Full HD 1080p, 4 target classes, 9 total detections with large clear two-wheelers and auto rickshaw)  
**Image Dimensions**: 1920x1080 (Aspect Ratio: 1.78:1)  

#### Ground Truth Statistics:
- **Filename**: `demo_05.jpg` (Original: `1181.jpg`)
- **Classes present**: Bus, Car, Motorcycle, Auto Rickshaw (4 distinct classes)
- **Vehicle count**: 8
- **Motorcycle count**: 2
- **Auto Rickshaw count**: 1
- **Bus count**: 1
- **Car count**: 4
- **Truck count**: 0
- **Largest box**: 219,968.0 px²
- **Smallest box**: 30,750.9 px²
- **Average box area**: 109,298.6625 px²
- **Why it is suitable for demonstration**: Dense Urban Multiclass Queue (Full HD 1080p, 4 target classes, 9 total detections with large clear two-wheelers and auto rickshaw). It provides exceptional visual clarity, representative bounding box scales for live visual demonstration, and tests multiple distinct vehicle geometries under standard surveillance viewpoints.

#### Model Verification (Frozen YOLOv8s @ conf=0.25, iou=0.50):
- **Ground-truth classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Predicted classes**: Bus, Car, Motorcycle, Auto Rickshaw
- **Total predictions**: 9
- **Per-class predictions**: Bus: 1, Car: 5, Motorcycle: 2, Auto Rickshaw: 1, Truck: 0
- **Detections and Confidence Values**:
  1. **Car** — Confidence: `0.9303` | Box `[x1=1587.9, y1=556.1, x2=1863.5, y2=737.3]` | Size: `275.5×181.2 px` (Area: 49,920.3 px²)
  2. **Car** — Confidence: `0.9297` | Box `[x1=1273.3, y1=593.5, x2=1676.5, y2=881.9]` | Size: `403.3×288.4 px` (Area: 116,306.0 px²)
  3. **Car** — Confidence: `0.9028` | Box `[x1=761.6, y1=590.0, x2=1108.9, y2=918.8]` | Size: `347.3×328.7 px` (Area: 114,161.5 px²)
  4. **Motorcycle** — Confidence: `0.8844` | Box `[x1=576.5, y1=675.2, x2=830.8, y2=1078.8]` | Size: `254.3×403.7 px` (Area: 102,642.5 px²)
  5. **Auto Rickshaw** — Confidence: `0.8760` | Box `[x1=160.7, y1=474.6, x2=662.7, y2=914.3]` | Size: `502.0×439.8 px` (Area: 220,748.0 px²)
  6. **Motorcycle** — Confidence: `0.8723` | Box `[x1=107.5, y1=721.2, x2=518.0, y2=1078.6]` | Size: `410.4×357.4 px` (Area: 146,696.0 px²)
  7. **Car** — Confidence: `0.8597` | Box `[x1=1165.2, y1=579.1, x2=1360.6, y2=741.7]` | Size: `195.4×162.7 px` (Area: 31,782.0 px²)
  8. **Bus** — Confidence: `0.6135` | Box `[x1=1495.2, y1=469.6, x2=1917.7, y2=657.8]` | Size: `422.5×188.2 px` (Area: 79,525.9 px²)
  9. **Car** — Confidence: `0.3101` | Box `[x1=1884.8, y1=550.0, x2=1920.0, y2=738.2]` | Size: `35.2×188.2 px` (Area: 6,628.5 px²)

---
