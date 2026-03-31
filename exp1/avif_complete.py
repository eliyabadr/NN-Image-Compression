import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
import pillow_avif  # Critical for AVIF support
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage # Added ToPILImage

# Importing from 'train' as per your manual edit
from train import SmallStudent 

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def run_avif_full_benchmark(epoch_num=30, max_images=10):
    device = 'cpu'
    print(f"Benchmarking on: {device}")

    # 1. Fetch from Kaggle
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model using your scratch path
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

    # Create directory for reconstructed outputs
    output_base_dir = "reconstructed_results"
    epoch_dir = os.path.join(output_base_dir, f"epoch_{epoch_num}")
    os.makedirs(epoch_dir, exist_ok=True)

    results = []

    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            
            # --- PHASE 0: Original Reference ---
            orig_img = Image.open(img_path).convert("RGB")
            x_orig = ToTensor()(orig_img).unsqueeze(0).to(device)

            # --- PHASE 1: AVIF Comparison ---
            temp_avif = "temp_input.avif"
            try:
                # Save as AVIF to get size and quality baseline
                orig_img.save(temp_avif, "AVIF", quality=60, speed=0)
                avif_size_kb = os.path.getsize(temp_avif) / 1024
                
                # Load AVIF to use as model input (bottleneck simulation)
                avif_img = Image.open(temp_avif).convert("RGB")
                w, h = avif_img.size
                x_avif = ToTensor()(avif_img).unsqueeze(0).to(device)
                psnr_avif = calculate_psnr(x_orig, x_avif)
            except Exception as e:
                print(f"Skipping {filename}: AVIF error ({e})")
                continue

            # --- PHASE 2: Model Compression & Timing ---
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            x_padded = F.pad(x_avif, (0, pad_w, 0, pad_h), "constant", 0)

            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time_ms = (time.perf_counter() - start_comp) * 1000

            tmp_bin = "temp_bits.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            model_size_kb = os.path.getsize(tmp_bin) / 1024

            start_decomp = time.perf_counter()
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            decomp_time_ms = (time.perf_counter() - start_decomp) * 1000
            
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            psnr_model = calculate_psnr(x_orig, x_hat)

            # --- PHASE 3: Save Reconstructed Image ---
            # Move tensor to CPU and convert back to PIL
            recon_pil = ToPILImage()(x_hat.squeeze(0).cpu())
            # Save as PNG to avoid further lossy compression artifacts
            recon_filename = f"recon_{os.path.splitext(filename)[0]}.png"
            recon_pil.save(os.path.join(epoch_dir, recon_filename))

            results.append({
                "file": filename,
                "avif_kb": avif_size_kb,
                "avif_psnr": psnr_avif,
                "model_kb": model_size_kb,
                "model_psnr": psnr_model,
                "comp_ms": comp_time_ms,
                "decomp_ms": decomp_time_ms
            })

    # 3. Final Comparison Summary
    df = pd.DataFrame(results)
    header = f"{'Filename':<15} | {'AVIF(KB)':<10} | {'A-PSNR':<8} | {'Model(KB)':<10} | {'M-PSNR':<8} | {'Enc(ms)':<8}"
    print("\n" + "="*95)
    print(f"RESULTS FOR EPOCH {epoch_num}")
    print(header)
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['file'][:15]:<15} | {r['avif_kb']:<10.1f} | {r['avif_psnr']:<8.2f} | {r['model_kb']:<10.1f} | {r['model_psnr']:<8.2f} | {r['comp_ms']:<8.1f}")
    
    print("="*95)
    print(f"AVG PSNR (Model vs Original): {df['model_psnr'].mean():.2f} dB")
    print(f"SIZE RATIO (AVIF/Model):      {df['avif_kb'].mean() / df['model_kb'].mean():.2f}:1")
    print(f"AVG ENCODE: {df['comp_ms'].mean():.1f} ms | AVG DECODE: {df['decomp_ms'].mean():.1f} ms")
    print(f"Images saved to: {epoch_dir}")

    # Cleanup temporary bitstreams
    for f in [temp_avif, tmp_bin]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Benchmarking across your requested epoch range
    for i in range(20, 21):
        run_avif_full_benchmark(epoch_num=i, max_images=1)
