# VisionToll: Experiment 4 — YOLOv5s Architecture Comparison
import sys
import time
import json
import torch
from pathlib import Path
import yolov5.train as train
import yolov5.val as val

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    ROOT = Path('D:/VisionToll')
    DATA_YAML = ROOT / 'data' / 'processed' / 'vision_toll_yolo' / 'data.yaml'
    OUTPUT_DIR = ROOT / 'results' / 'experiments'
    EXP_NAME = 'vision_toll_exp4_yolov5s'
    MODEL_SAVE_DIR = ROOT / 'models' / 'exp4_yolov5s'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    CLASS_NAMES = {0: 'Bus', 1: 'Car', 2: 'Motorcycle', 3: 'Auto Rickshaw', 4: 'Truck'}

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    cuda_ver = torch.version.cuda or 'N/A'
    print('VisionToll Exp4: YOLOv5s Architecture Comparison')
    print(f'Device: {device_name}  CUDA: {cuda_ver}  PyTorch: {torch.__version__}')

    weights_path = ROOT / 'yolov5s.pt'
    if not weights_path.exists():
        print(f"Weights not found locally at {weights_path}, YOLOv5 will auto-download.")
    
    t0 = time.time()
    
    # YOLOv5 training uses argparse config object, but can be invoked via train.run
    print("\nStarting YOLOv5s Training...")
    try:
        train.run(
            data=str(DATA_YAML),
            epochs=50,
            imgsz=640,
            batch_size=16,
            project=str(OUTPUT_DIR),
            name=EXP_NAME,
            weights='yolov5s.pt',
            device='0',
            workers=4,
            seed=42,
            exist_ok=False
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return

    duration = time.time() - t0
    print(f'\nTraining done {duration/60:.2f} min')

    run_dir = OUTPUT_DIR / EXP_NAME
    best_pt = run_dir / 'weights' / 'best.pt'
    last_pt = run_dir / 'weights' / 'last.pt'
    
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, MODEL_SAVE_DIR / 'best.pt')
        if last_pt.exists():
            shutil.copy2(last_pt, MODEL_SAVE_DIR / 'last.pt')
        print(f"Saved best model to {MODEL_SAVE_DIR / 'best.pt'}")
    else:
        print("Error: best.pt not found after training!")
        return

    # Validation
    print("\nStarting Validation...")
    val_metrics = val.run(
        data=str(DATA_YAML),
        weights=str(best_pt),
        batch_size=16,
        imgsz=640,
        device='0',
        workers=4,
        task='val',
        plots=True,
        project=str(OUTPUT_DIR),
        name=EXP_NAME + '_val',
        exist_ok=True
    )
    
    # val_metrics returns a tuple: (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    # Or in newer yolov5, (mp, mr, map50, map50_95) as first element if using dict/list
    # Let's run evaluation using YOLO() if possible? YOLO() is ultralytics v8.
    # YOLOv5 uses a different return format. We'll parse it.
    
    print("\nExtracting metrics...")

if __name__ == '__main__':
    main()
