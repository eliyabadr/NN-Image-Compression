# Updated File version 2 (Includes WebP Benchmark)
import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
from PIL import Image
from torchvision.transforms import ToTensor
from train import SmallStudent 

def run_split_latency_benchmark(epoch_num=15, max_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    # 1. Locate Dataset
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model
    img_count = 60000 if epoch_num >= 10 else 40000
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_imgs_{img_count}.pth'
    
    model = SmallStudent(N=64, M=96).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)

    supported_ext = ('.jpg', '.jpeg', '.png')
    all_files = [os.path.join(r, f) for r, _, fs in os.walk(dataset_path) for f in fs if f.lower().endswith(supported_ext)]
    image_paths = all_files[:max_images]

    results = []

    with torch.no_grad():
        for img_path in image_paths:
            # --- Load Image & Sizes ---
            img = Image.open(img_path).convert("RGB")
            
            # 1. Original JPEG/Source Size
            orig_size_kb = os.path.getsize(img_path) / 1024

            # 2. WebP Size (Standard Quality 75)
            temp_webp = "temp_bench.webp"
            img.save(temp_webp, "WEBP", quality=75)
            webp_size_kb = os.path.getsize(temp_webp) / 1024

            # --- Model Compression ---
            w, h = img.size
            x = ToTensor()(img).unsqueeze(0).to(device)
            p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
            x_padded = F.pad(x, p2d, "constant", 0)

            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time = (time.perf_counter() - start_comp) * 1000

            tmp_bin = "temp_bitstream.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            bitstream_size_kb = os.path.getsize(tmp_bin) / 1024

            # --- Model Decompression ---
            start_decomp = time.perf_counter()
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            decomp_time = (time.perf_counter() - start_decomp) * 1000
            
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            
            # PSNR
            mse = F.mse_loss(x, x_hat).item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100

            results.append({
                "file": os.path.basename(img_path),
                "psnr": psnr,
                "comp_ms": comp_time,
                "decomp_ms": decomp_time,
                "orig_kb": orig_size_kb,
                "webp_kb": webp_size_kb,
                "bitstream_kb": bitstream_size_kb
            })

    # Cleanup temp files
    if os.path.exists("temp_bitstream.bin"): os.remove("temp_bitstream.bin")
    if os.path.exists("temp_bench.webp"): os.remove("temp_bench.webp")

    # 4. Display Results
    df = pd.DataFrame(results)
    header = f"{'Filename':<15} | {'PSNR':<5} | {'Orig(KB)':<8} | {'WebP(KB)':<8} | {'Bits(KB)':<8}"
    print("\n" + "="*70)
    print(header)
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['file'][:15]:<15} | {r['psnr']:<5.1f} | {r['orig_kb']:<8.1f} | {r['webp_kb']:<8.1f} | {r['bitstream_kb']:<8.1f}")
    
    print("="*70)
    print(f"AVERAGE PSNR: {df['psnr'].mean():.2f} dB")
    print(f"AVERAGE MODEL RATIO (Orig/Bits): {df['orig_kb'].mean() / df['bitstream_kb'].mean():.2f}:1")
    print(f"AVERAGE WEBP RATIO (Orig/WebP): {df['orig_kb'].mean() / df['webp_kb'].mean():.2f}:1")

if __name__ == "__main__":
    run_split_latency_benchmark(epoch_num=15, max_images=10)
