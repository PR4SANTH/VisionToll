import os
import sys
import time
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
EXP_DIR = ROOT / "results" / "experiments" / "vision_toll_exp5_faster_rcnn"
EXP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR = ROOT / "models" / "exp5_faster_rcnn"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 0 is background, 1-5 are classes
CLASS_NAMES = {1: "Bus", 2: "Car", 3: "Motorcycle", 4: "Auto Rickshaw", 5: "Truck"}
NUM_CLASSES = 6  # 5 + background

class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=(640, 640)):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.images = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        im0 = cv2.imread(str(img_path))
        orig_h, orig_w = im0.shape[:2]
        
        im = cv2.resize(im0, self.img_size)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = torch.from_numpy(im.transpose((2, 0, 1))).float() / 255.0

        boxes = []
        labels = []

        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Shift class +1 for background
                    labels.append(c + 1)
                    
                    # Convert to absolute coords [xmin, ymin, xmax, ymax] for resized image
                    abs_xc = xc * self.img_size[0]
                    abs_yc = yc * self.img_size[1]
                    abs_w = w * self.img_size[0]
                    abs_h = h * self.img_size[1]
                    
                    xmin = max(0, abs_xc - abs_w / 2)
                    ymin = max(0, abs_yc - abs_h / 2)
                    xmax = min(self.img_size[0], abs_xc + abs_w / 2)
                    ymax = min(self.img_size[1], abs_yc + abs_h / 2)
                    
                    # Ensure positive area
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return im_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("=" * 50)
    print("EXPERIMENT 5 — FASTER R-CNN ResNet50-FPN")
    print("=" * 50)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Build model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=640, max_size=640)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # DataLoader
    train_dataset = YoloToFasterRCNNDataset(DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    val_dataset = YoloToFasterRCNNDataset(DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")
    
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    
    batch_size = 4
    print(f"Starting with batch size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    epochs = 50

    start_epoch = 0
    if (EXP_DIR / "last.pt").exists():
        print("Found last.pt, resuming training...")
        model.load_state_dict(torch.load(EXP_DIR / "last.pt"))
        # We assume we crashed after saving last.pt but before completing the epoch fully.
        # However, to be safe and simple, let's just assume we resume from epoch 10 if last.pt exists.
        # Actually, let's just let it train the remaining epochs. 
        # A simple hack for this exact situation:
        start_epoch = 10
        print(f"Resuming from epoch {start_epoch+1}")

    t0 = time.time()
    scaler = torch.amp.GradScaler('cuda')
    
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    best_map50 = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), EXP_DIR / "last.pt")
        
        # Evaluate every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            model.eval()
            metric = MeanAveragePrecision(class_metrics=True)
            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = list(img.to(device) for img in val_images)
                    preds = model(val_images)
                    
                    # Convert to CPU for metrics
                    metric_preds = []
                    for p in preds:
                        metric_preds.append({
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu()
                        })
                    
                    metric_targets = []
                    for t in val_targets:
                        metric_targets.append({
                            "boxes": t["boxes"].cpu(),
                            "labels": t["labels"].cpu()
                        })
                        
                    metric.update(metric_preds, metric_targets)
                    
            results = metric.compute()
            map50 = results['map_50'].item()
            print(f"Validation mAP@50: {map50:.4f}, mAP@50-95: {results['map'].item():.4f}")
            
            if map50 > best_map50:
                best_map50 = map50
                torch.save(model.state_dict(), EXP_DIR / "best.pt")
                print(f"Saved new best.pt (mAP50: {best_map50:.4f})")

    duration = time.time() - t0
    import shutil
    shutil.copy2(EXP_DIR / "best.pt", MODEL_SAVE_DIR / "best.pt")

    print(f"Training complete in {duration/60:.2f} minutes.")
    print("Saved to", MODEL_SAVE_DIR / "best.pt")

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
