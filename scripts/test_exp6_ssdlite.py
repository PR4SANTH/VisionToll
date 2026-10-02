import os
import time
import torch
import torch.optim as optim
from pathlib import Path
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

from exp6_ssdlite_utils import VisionTollSSDLiteDataset, collate_fn, save_checkpoint, load_checkpoint

ROOT = Path("D:/VisionToll")
DATA_DIR = ROOT / "data" / "processed" / "vision_toll_yolo"

def test_dataset_and_classes():
    print("Testing dataset loading and class conversion...")
    dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    assert len(dataset) > 0, "Train dataset is empty!"
    
    # Check first 5 images for valid outputs
    empty_label_seen = False
    valid_label_seen = False
    
    for i in range(min(50, len(dataset))):
        img, target = dataset[i]
        assert img.shape[0] == 3
        assert img.shape[1] == 320
        assert img.shape[2] == 320
        
        boxes = target["boxes"]
        labels = target["labels"]
        areas = target["area"]
        
        if len(labels) == 0:
            empty_label_seen = True
            assert len(boxes) == 0
            assert len(areas) == 0
        else:
            valid_label_seen = True
            # Validate classes are 1..5
            assert torch.all(labels >= 1) and torch.all(labels <= 5), f"Found invalid label {labels}"
            # Validate boxes
            assert torch.all(boxes[:, 2] > boxes[:, 0]), "Box xmax <= xmin"
            assert torch.all(boxes[:, 3] > boxes[:, 1]), "Box ymax <= ymin"
            assert torch.all(boxes >= 0.0) and torch.all(boxes <= 320.0), "Box coords out of bounds"
            
    print("Dataset test passed.")
    print(f"Empty labels verified? {empty_label_seen}")
    print(f"Valid labels verified? {valid_label_seen}")

def test_model_construction():
    print("Testing model construction and head replacement...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    pretrained_model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    target_model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=6)
    
    state_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if "head.classification_head" not in k}
    
    missing, unexpected = target_model.load_state_dict(filtered_state_dict, strict=False)
    
    assert len(unexpected) == 0, "Unexpected keys found"
    assert all("head.classification_head" in k for k in missing), "Missing non-classification keys!"
    print("Model construction test passed.")
    return target_model

def test_cuda_smoke_and_checkpointing(model):
    print("Testing CUDA, memory, loss calculation, and checkpointing...")
    device = torch.device('cuda')
    assert torch.cuda.is_available(), "CUDA not available!"
    model.to(device)
    
    dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    img, target = dataset[0]
    
    # Need to simulate a batch of at least 2 for robust testing, let's just use 4
    batch_size = 4
    images = [img.to(device)] * batch_size
    targets = [{k: v.to(device) for k, v in target.items()} for _ in range(batch_size)]
    
    # For training SSD, the model expects lists of images and dicts of targets
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    torch.cuda.reset_peak_memory_stats()
    
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    assert losses > 0, "Loss is not positive"
    
    losses.backward()
    optimizer.step()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM during batch=4 forward+backward pass: {peak_vram:.2f} MB")
    
    # Test Checkpointing
    print("Testing checkpointing...")
    save_checkpoint("scratch/test_checkpoint.pt", 1, model, optimizer, None, None, {"map50": 0.5}, {"batch": 4})
    
    # Reset model and load
    model_new = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=6)
    optimizer_new = optim.AdamW(model_new.parameters(), lr=0.01) # different LR to verify loading
    epoch, metrics = load_checkpoint("scratch/test_checkpoint.pt", model_new, optimizer_new)
    
    assert epoch == 1
    assert metrics["map50"] == 0.5
    # Verify optimizer state loaded
    assert optimizer_new.param_groups[0]['lr'] == 0.001
    print("Checkpointing test passed.")

def test_tiny_overfit(model):
    print("Running tiny overfit test...")
    device = torch.device('cuda')
    model.to(device)
    model.train()
    
    dataset = VisionTollSSDLiteDataset(DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    
    # Grab 4 images with non-empty targets
    images, targets = [], []
    for i in range(len(dataset)):
        img, tgt = dataset[i]
        if len(tgt["labels"]) > 0:
            images.append(img.to(device))
            targets.append({k: v.to(device) for k, v in tgt.items()})
        if len(images) == 4:
            break
            
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    initial_loss = None
    final_loss = None
    for i in range(20):
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        if i == 19:
            final_loss = loss.item()
            
    print(f"Tiny overfit: Initial loss = {initial_loss:.4f}, Final loss = {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during overfit test"
    
    print("Overfit test passed.")

if __name__ == "__main__":
    try:
        test_dataset_and_classes()
        model = test_model_construction()
        test_cuda_smoke_and_checkpointing(model)
        test_tiny_overfit(model)
        print("ALL TESTS PASSED SUCCESSFULLY.")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
