import json, os, sys
import torch
import torchvision

def main():
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version.split('\n')[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024),
        })
    os.makedirs(r"D:/VisionToll/results/preflight/vision_toll_net", exist_ok=True)
    with open(r"D:/VisionToll/results/preflight/vision_toll_net/environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))

if __name__ == "__main__":
    main()
