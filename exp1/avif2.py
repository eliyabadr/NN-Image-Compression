import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
import pillow_avif 
from PIL import Image
from torchvision.transforms import ToTensor

# Importing from your specific file
from train import SmallStudent 

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def run_avif_conversion_benchmark(epoch_num=30, max_images=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    # 1. Fetch from Kaggle
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model using your specific scratch path
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_compression_focus.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

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
            filename = os.path.basename(img_path)
            
            # --- PHASE 0: Original Reference ---
            orig_img = Image.open(img_path).convert("RGB")
            x_orig = ToTensor()(orig_img).unsqueeze(0).to(device)

            # --- PHASE 1: AVIF Bottleneck ---
            temp_avif = "temp_input.avif"
            try:
                orig_img.save(temp_avif, "AVIF", quality=60, speed=0)
            except Exception: continue

            if not os.path.exists(temp_avif): continue
            avif_size_kb = os.path.getsize(temp_avif) / 1024
            
            avif_img = Image.open(temp_avif).convert("RGB")
            w, h = avif_img.size
            x_avif = ToTensor()(avif_img).unsqueeze(0).to(device)

            # --- PHASE 2: Neural Compression & Timing ---
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            x_padded = F.pad(x_avif, (0, pad_w, 0, pad_h), "constant", 0)

            # Time Compression (Encoding)
            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time_ms = (time.perf_counter() - start_comp) * 1000

            # Measure Bitstream
            tmp_bin = "temp_bits.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            model_size_kb = os.path.getsize(tmp_bin) / 1024

            # Time Decompression (Decoding)
            start_decomp = time.perf_counter()
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            decomp_time_ms = (time.perf_counter() - start_decomp) * 1000
            
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            psnr_val = calculate_psnr(x_orig, x_hat)

            results.append({
                "file": filename,
                "psnr": psnr_val,
                "avif_kb": avif_size_kb,
                "model_kb": model_size_kb,
                "comp_ms": comp_time_ms,
                "decomp_ms": decomp_time_ms
            })

    # 3. Final Summary Table
    df = pd.DataFrame(results)
    header = f"{'Filename':<15} | {'PSNR':<5} | {'Model(KB)':<9} | {'Comp(ms)':<9} | {'Decomp(ms)':<10}"
    print("\n" + "="*70)
    print(header)
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['file'][:15]:<15} | {r['psnr']:<5.1f} | {r['model_kb']:<9.1f} | {r['comp_ms']:<9.1f} | {r['decomp_ms']:<10.1f}")
    
    print("="*70)
    print(f"AVG COMPRESSION TIME:   {df['comp_ms'].mean():.2f} ms")
    print(f"AVG DECOMPRESSION TIME: {df['decomp_ms'].mean():.2f} ms")
    print(f"AVG PSNR: {df['psnr'].mean():.2f} dB")

    # Cleanup
    for f in [temp_avif, tmp_bin]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    run_avif_conversion_benchmark(epoch_num=30, max_images=1)
