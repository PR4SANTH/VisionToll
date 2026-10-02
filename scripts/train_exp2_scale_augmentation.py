# VisionToll Experiment 2 -- YOLOv8n Scale-Augmentation Intervention
# Intervention: copy_paste=0.3, scale=0.9 (Exp1 defaults: 0.0, 0.5)
# Evidence: Motorcycle mAP50-95=0.5995 (primary bottleneck); 106 small-vehicle val images
# All other hyperparameters identical to Experiment 1.
# Test set: NOT used -- validation split only.

import sys, time, json, shutil, csv, torch
from pathlib import Path
from ultralytics import YOLO


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    ROOT           = Path('D:/VisionToll')
    DATA_YAML      = ROOT / 'data' / 'processed' / 'vision_toll_yolo' / 'data.yaml'
    OUTPUT_DIR     = ROOT / 'results' / 'experiments'
    EXP_NAME       = 'vision_toll_exp2_scale_augmentation'
    MODEL_SAVE_DIR = ROOT / 'models' / 'exp2_scale_augmentation'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    CLASS_NAMES = {0: 'Bus', 1: 'Car', 2: 'Motorcycle', 3: 'Auto Rickshaw', 4: 'Truck'}

    EXP1 = {'precision': 0.8579, 'recall': 0.7819, 'f1': 0.8181, 'map50': 0.8655, 'map50_95': 0.7197}
    EXP1_PC = {
        'Bus':          {'precision': 0.857, 'recall': 0.729, 'map50': 0.784},
        'Car':          {'precision': 0.896, 'recall': 0.818, 'map50': 0.925},
        'Motorcycle':   {'precision': 0.812, 'recall': 0.763, 'map50': 0.839},
        'Auto Rickshaw':{'precision': 0.886, 'recall': 0.812, 'map50': 0.909},
        'Truck':        {'precision': 0.840, 'recall': 0.787, 'map50': 0.871},
    }

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    cuda_ver    = torch.version.cuda or 'N/A'
    print('VisionToll Exp2: copy_paste=0.3 scale=0.9')
    print(f'Device: {device_name}  CUDA: {cuda_ver}  PyTorch: {torch.__version__}')

    pretrained = ROOT / 'models' / 'baseline' / 'yolov8n.pt'
    assert pretrained.exists(), f'Missing: {pretrained}'
    model = YOLO(str(pretrained))
    t0 = time.time()

    model.train(
        data=str(DATA_YAML),
        epochs=50, imgsz=640, batch=16, patience=15, seed=42,
        project=str(OUTPUT_DIR), name=EXP_NAME,
        pretrained=True, workers=0, plots=True, save=True,
        scale=0.9,
        copy_paste=0.3,
        close_mosaic=10, mosaic=1.0,
        flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, shear=0.0, perspective=0.0,
        mixup=0.0, cutmix=0.0, erasing=0.4,
    )

    duration = time.time() - t0
    print(f'Training done {duration/60:.2f} min')

    run_dir = OUTPUT_DIR / EXP_NAME
    best_pt = run_dir / 'weights' / 'best.pt'
    last_pt = run_dir / 'weights' / 'last.pt'
    shutil.copy2(best_pt, MODEL_SAVE_DIR / 'best.pt')
    shutil.copy2(last_pt, MODEL_SAVE_DIR / 'last.pt')

    ev = YOLO(str(best_pt))
    vm = ev.val(data=str(DATA_YAML), split='val', batch=16, imgsz=640, device=0, plots=True, workers=0)

    mp    = float(vm.box.mp)
    mr    = float(vm.box.mr)
    map50 = float(vm.box.map50)
    m5095 = float(vm.box.map)
    f1    = 2 * mp * mr / max(mp + mr, 1e-6)

    print(f'Precision  {mp:.4f}  delta={mp - EXP1["precision"]:+.4f}')
    print(f'Recall     {mr:.4f}  delta={mr - EXP1["recall"]:+.4f}')
    print(f'mAP50      {map50:.4f}  delta={map50 - EXP1["map50"]:+.4f}')
    print(f'mAP50-95   {m5095:.4f}  delta={m5095 - EXP1["map50_95"]:+.4f}')

    pcc = {}
    for cid in range(5):
        cn = CLASS_NAMES[cid]
        p2 = float(vm.box.p[cid])    if cid < len(vm.box.p)    else 0.0
        r2 = float(vm.box.r[cid])    if cid < len(vm.box.r)    else 0.0
        m2 = float(vm.box.maps[cid]) if cid < len(vm.box.maps) else 0.0
        f2 = 2 * p2 * r2 / max(p2 + r2, 1e-6)
        e1 = EXP1_PC[cn]
        pcc[cn] = {'precision': round(p2,4), 'recall': round(r2,4), 'f1': round(f2,4), 'map50': round(m2,4)}
        print(f'  {cn}: P={p2:.4f}(d{p2 - e1["precision"]:+.4f}) R={r2:.4f}(d{r2 - e1["recall"]:+.4f}) mAP50={m2:.4f}(d{m2 - e1["map50"]:+.4f})')

    rpt = {
        'experiment': 'Exp2 YOLOv8n Scale-Augmentation',
        'intervention': {'scale': 0.9, 'copy_paste': 0.3, 'exp1_scale': 0.5, 'exp1_copy_paste': 0.0},
        'device': device_name, 'pytorch': torch.__version__, 'cuda': cuda_ver,
        'epochs': 50, 'batch': 16, 'imgsz': 640, 'seed': 42,
        'duration_s': round(duration, 2), 'duration_min': round(duration / 60, 2),
        'metrics': {'precision': round(mp,4), 'recall': round(mr,4), 'f1': round(f1,4),
                    'map50': round(map50,4), 'map50_95': round(m5095,4)},
        'per_class': pcc,
        'vs_exp1': {
            'delta_precision': round(mp - EXP1['precision'], 4),
            'delta_recall':    round(mr - EXP1['recall'], 4),
            'delta_map50':     round(map50 - EXP1['map50'], 4),
            'delta_map50_95':  round(m5095 - EXP1['map50_95'], 4),
        }
    }

    rp = run_dir / 'exp2_validation_results.json'
    json.dump(rpt, open(rp, 'w', encoding='utf-8'), indent=2)
    print(f'Report: {rp}')

    exp_csv = OUTPUT_DIR / 'experiments.csv'
    FN = ['Experiment_ID','Model','Intervention','Precision','Recall','F1','mAP50','mAP50-95','Training_Time_min','Key_Observation']
    nr = {'Experiment_ID': 'Exp 2', 'Model': 'YOLOv8n', 'Intervention': 'copy_paste=0.3 scale=0.9',
          'Precision': round(mp,4), 'Recall': round(mr,4), 'F1': round(f1,4),
          'mAP50': round(map50,4), 'mAP50-95': round(m5095,4),
          'Training_Time_min': round(duration/60,2),
          'Key_Observation': 'Scale-aug targeting Motorcycle small-obj weakness'}
    rows = []
    if exp_csv.exists():
        for r in csv.DictReader(open(exp_csv, encoding='utf-8')):
            if r.get('Experiment_ID') != 'Exp 2':
                rows.append(r)
    rows.append(nr)
    with open(exp_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FN)
        w.writeheader()
        w.writerows(rows)
    print('EXP2 COMPLETE. Test set untouched.')


if __name__ == '__main__':
    main()
