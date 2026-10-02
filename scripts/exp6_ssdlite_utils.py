import os
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path

class VisionTollSSDLiteDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=(320, 320)):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.images = sorted([p for p in self.img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png"]])
        
        # Verify train/val separation by ensuring lbl_dir contains expected text files
        # The exact data leakage check is handled at the experiment level.
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        im0 = cv2.imread(str(img_path))
        if im0 is None:
            raise ValueError(f"Failed to read image {img_path}")
            
        im = cv2.resize(im0, self.img_size)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = torch.from_numpy(im.transpose((2, 0, 1))).float() / 255.0

        boxes = []
        labels = []
        areas = []

        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    # YOLO classes are 0..4, Torchvision requires foreground classes to start at 1 (0 is background)
                    torchvision_class = c + 1
                    
                    # Validate class 1..5
                    if torchvision_class < 1 or torchvision_class > 5:
                        continue
                        
                    # YOLO: xc, yc, w, h (normalized)
                    # Convert to absolute coords [xmin, ymin, xmax, ymax]
                    abs_xc = xc * self.img_size[0]
                    abs_yc = yc * self.img_size[1]
                    abs_w = w * self.img_size[0]
                    abs_h = h * self.img_size[1]
                    
                    xmin = max(0.0, abs_xc - abs_w / 2.0)
                    ymin = max(0.0, abs_yc - abs_h / 2.0)
                    xmax = min(float(self.img_size[0]), abs_xc + abs_w / 2.0)
                    ymax = min(float(self.img_size[1]), abs_yc + abs_h / 2.0)
                    
                    # Validate box
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(torchvision_class)
                        areas.append((xmax - xmin) * (ymax - ymin))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd
        }

        return im_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def save_checkpoint(filepath, epoch, model, optimizer, scheduler, scaler, metrics, config, model_name="ssdlite320_mobilenet_v3_large"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_metric": metrics,
        "class_names": {0: "background", 1: "Bus", 2: "Car", 3: "Motorcycle", 4: "Auto Rickshaw", 5: "Truck"},
        "config": config,
        "model_name": model_name
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", {})
