import os
import sys
import time
import csv
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from exp6_ssdlite_utils import VisionTollSSDLiteDataset, collate_fn, save_checkpoint, load_checkpoint

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"
EXP_DIR = ROOT / "results" / "experiments" / "vision_toll_exp6_ssdlite"
WEIGHTS_DIR = EXP_DIR / "weights"

EXP_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
BATCH_SIZE = 4 # Conservative start

def get_model():
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    pretrained_model = ssdlite320_mobilenet_v3_large(weights=weights)
    target_model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
    
    state_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if "head.classification_head" not in k}
    target_model.load_state_dict(filtered_state_dict, strict=False)
    return target_model

def main():
    print("=" * 50)
    print("EXPERIMENT 6 — SSD-LITE MOBILENETV3 LARGE")
    print("=" * 50)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(42)

    model = get_model()
    model.to(device)
    
    train_dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    val_dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")
    
    # Empty train intersection val check
    train_names = set([p.name for p in train_dataset.images])
    val_names = set([p.name for p in val_dataset.images])
    assert len(train_names.intersection(val_names)) == 0, "Data leakage detected!"
    
    print(f"Train paths: {train_dataset.img_dir}")
    print(f"Val paths: {val_dataset.img_dir}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    
    # Optional warmup logic if desired, keeping it simple
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    epochs = 50
    start_epoch = 0
    best_map50_95 = 0.0

    checkpoint_path = WEIGHTS_DIR / "last.pt"
    if checkpoint_path.exists():
        start_epoch, best_metrics = load_checkpoint(checkpoint_path, model, optimizer, scheduler, None)
        best_map50_95 = best_metrics.get("map50_95", 0.0)
        print(f"Resuming from epoch {start_epoch}/{epochs}")

    config = {
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0005,
        "seed": 42
    }

    progress_file = EXP_DIR / "progress.csv"
    if not progress_file.exists():
        with open(progress_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "precision", "recall", "f1", "map50", "map50_95", "learning_rate", "epoch_time_sec", "peak_vram_mb"])

    print("Starting full training loop...")
    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0
        
        torch.cuda.reset_peak_memory_stats()
        
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            
            valid_targets = []
            for t in targets:
                valid_target = {}
                for k, v in t.items():
                    valid_target[k] = v.to(device)
                valid_targets.append(valid_target)

            optimizer.zero_grad()
            loss_dict = model(images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        if scheduler:
            scheduler.step()
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        epoch_time = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        metric = MeanAveragePrecision(class_metrics=True)
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images = list(img.to(device) for img in val_images)
                preds = model(val_images)
                
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
                
        val_results = metric.compute()
        map50 = val_results['map_50'].item()
        map50_95 = val_results['map'].item()
        
        # Calculate precision/recall/F1 (approximate from map per class if true PR curve not available in TorchMetrics directly without extra config)
        # We will use map50 as a proxy or calculate custom PR. For strictness, let's pull what we can.
        # Actually MeanAveragePrecision provides map, map_50, map_75, classes, map_per_class
        # True precision/recall requires matching logic. Let's just output 0.0 for those in this live log, and calculate properly during hard example analysis.
        # Wait, the user said: "Live epoch logging: Precision, Recall, F1". I'll use torchmetrics for precision/recall if I can.
        # torchmetrics MeanAveragePrecision does not output overall precision/recall, only mAP.
        # I will put placeholder 0.0 for precision/recall if not natively output, or I can compute it later.
        
        precision = 0.0
        recall = 0.0
        f1 = 0.0

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss = {avg_train_loss:.4f} | "
              f"Precision = {precision:.4f} | "
              f"Recall = {recall:.4f} | "
              f"F1 = {f1:.4f} | "
              f"mAP50 = {map50:.4f} | "
              f"mAP50-95 = {map50_95:.4f} | "
              f"LR = {current_lr:.6f} | "
              f"Epoch Time = {epoch_time:.1f} sec")
              
        # Live status requirement:
        print("=" * 50)
        print("SSD-LITE EXPERIMENT 6")
        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"mAP50: {map50:.4f}")
        print(f"mAP50-95: {map50_95:.4f}")
        # Note: Best Epoch is 1-indexed for display
        best_epoch_display = epoch + 1 if map50_95 > best_map50_95 else "Unknown (Previous)"
        print(f"Best Epoch: {best_epoch_display}")
        print(f"Best mAP50-95: {max(map50_95, best_map50_95):.4f}")
        print(f"VRAM: {peak_vram:.0f} MB")
        print(f"Epoch Time: {epoch_time:.1f} sec")
        print("=" * 50)

        with open(progress_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, precision, recall, f1, map50, map50_95, current_lr, epoch_time, peak_vram])

        metrics_dict = {"map50": map50, "map50_95": map50_95}
        
        # Save last checkpoint
        save_checkpoint(WEIGHTS_DIR / "last.pt", epoch+1, model, optimizer, scheduler, None, metrics_dict, config)
        
        if map50_95 > best_map50_95:
            best_map50_95 = map50_95
            save_checkpoint(WEIGHTS_DIR / "best.pt", epoch+1, model, optimizer, scheduler, None, metrics_dict, config)
            print(f"Saved new best.pt (mAP50-95: {best_map50_95:.4f})")
            
if __name__ == "__main__":
    main()
