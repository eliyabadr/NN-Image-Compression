import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

# 1. Registration & Import Check
try:
    import pillow_avif
    print("✓ AVIF Plugin loaded.")
except ImportError:
    print("! Warning: pillow-avif-plugin not found. Ensure it is installed in your current environment.")

# Import your model definition
from train import SmallStudent 

def calculate_psnr(img1_tensor, img2_tensor):
    mse = F.mse_loss(img1_tensor, img2_tensor).item()
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def run_avif_conversion_benchmark(epoch_num=30, max_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    # 1. Fetch Dataset from Kaggle
    print("Downloading/Locating Kaggle dataset...")
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
            
            # --- PHASE 0: Load Original for PSNR baseline ---
            orig_img = Image.open(img_path).convert("RGB")
            x_orig = ToTensor()(orig_img).unsqueeze(0).to(device)

            # --- PHASE 1: Convert Kaggle Source to AVIF ---
            temp_avif = "temp_input.avif"
            try:
                # We use quality 60 (standard high-efficiency target)
                orig_img.save(temp_avif, "AVIF", quality=60, speed=0)
            except Exception as e:
                print(f"Error: Failed to save AVIF for {filename}. Details: {e}")
                continue # Skips to next image instead of crashing

            # Check if file exists before calling getsize
            if not os.path.exists(temp_avif):
                print(f"Error: {temp_avif} was not created for {filename}")
                continue

            avif_size_kb = os.path.getsize(temp_avif) / 1024
            
            # Load AVIF to see what quality it provided
            avif_img = Image.open(temp_avif).convert("RGB")
            x_avif = ToTensor()(avif_img).unsqueeze(0).to(device)
            psnr_avif_vs_orig = calculate_psnr(x_orig, x_avif)

            # --- PHASE 2: Input AVIF into SmallStudent ---
            w, h = avif_img.size
            # Padding to multiple of 64
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            x_padded = F.pad(x_avif, (0, pad_w, 0, pad_h), "constant", 0)

            # Model Compression
            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time = (time.perf_counter() - start_comp) * 1000

            # Measure Bitstream Size
            tmp_bin = "temp_bits.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            model_size_kb = os.path.getsize(tmp_bin) / 1024

            # Decompress
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            
            # Quality of Model vs the AVIF input
            psnr_model_vs_avif = calculate_psnr(x_avif, x_hat)

            results.append({
                "file": filename,
                "avif_kb": avif_size_kb,
                "avif_psnr": psnr_avif_vs_orig,
                "model_kb": model_size_kb,
                "model_psnr": psnr_model_vs_avif
            })

    # 3. Final Summary & Table
    df = pd.DataFrame(results)
    header = f"{'Filename':<15} | {'AVIF(KB)':<9} | {'AVIF PSNR':<9} | {'Model(KB)':<9} | {'Model PSNR':<9}"
    print("\n" + "="*80)
    print(header)
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['file'][:15]:<15} | {r['avif_kb']:<9.1f} | {r['avif_psnr']:<9.2f} | {r['model_kb']:<9.1f} | {r['model_psnr']:<9.2f}")
    
    print("="*80)
    print(f"AVG AVIF:  {df['avif_kb'].mean():.1f} KB  | {df['avif_psnr'].mean():.2f} dB")
    print(f"AVG MODEL: {df['model_kb'].mean():.1f} KB  | {df['model_psnr'].mean():.2f} dB")
    print(f"RATIO (AVIF/Model): {df['avif_kb'].mean() / df['model_kb'].mean():.2f}:1")

    # Cleanup
    for f in [temp_avif, tmp_bin]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Matches your manual note for Epoch 30
    run_avif_conversion_benchmark(epoch_num=25, max_images=1)
