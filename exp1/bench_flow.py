#Version 1
import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pandas as pd
import kagglehub
import io
from PIL import Image
from torchvision.transforms import ToTensor
from train import SmallStudent 

def run_webp_to_model_benchmark(epoch_num=15, max_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model (using your existing SmallStudent definition)
    img_count = 60000 if epoch_num >= 10 else 40000
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_imgs_{img_count}.pth'
    model = SmallStudent(N=64, M=96).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)

    # 2. Locate Dataset
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    supported_ext = ('.jpg', '.jpeg', '.png')
    all_files = [os.path.join(r, f) for r, _, fs in os.walk(dataset_path) for f in fs if f.lower().endswith(supported_ext)]
    image_paths = all_files[:max_images]

    results = []

    with torch.no_grad():
        for img_path in image_paths:
            # --- STEP 1: LOAD ORIGINAL JPG ---
            orig_jpg_size = os.path.getsize(img_path) / 1024
            img = Image.open(img_path).convert("RGB")

            # --- STEP 2: CONVERT TO WebP (The "New" Source) ---
            # We save to a buffer to simulate the file existing
            webp_buffer = io.BytesIO()
            img.save(webp_buffer, format="WEBP", quality=75)
            webp_size_kb = len(webp_buffer.getvalue()) / 1024
            
            # --- STEP 3: DECOMPRESS WebP FOR MODEL ---
            webp_buffer.seek(0)
            img_from_webp = Image.open(webp_buffer).convert("RGB")
            
            # --- STEP 4: MODEL COMPRESSION ---
            w, h = img_from_webp.size
            x = ToTensor()(img_from_webp).unsqueeze(0).to(device)
            p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
            x_padded = F.pad(x, p2d, "constant", 0)

            start_comp = time.perf_counter()
            compressed_data = model.compress(x_padded) 
            comp_time = (time.perf_counter() - start_comp) * 1000

            # Measure Bitstream Size
            tmp_bin = "temp_bitstream.bin"
            with open(tmp_bin, "wb") as f:
                pickle.dump(compressed_data, f)
            bitstream_size_kb = os.path.getsize(tmp_bin) / 1024

            # Model Decompression for PSNR
            out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
            x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
            mse = F.mse_loss(x, x_hat).item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100

            results.append({
                "file": os.path.basename(img_path),
                "psnr": psnr,
                "orig_jpg_kb": orig_jpg_size,
                "webp_kb": webp_size_kb,
                "bitstream_kb": bitstream_size_kb
            })

    # Cleanup
    if os.path.exists("temp_bitstream.bin"): os.remove("temp_bitstream.bin")

    # 5. Display Results
    df = pd.DataFrame(results)
    header = f"{'Filename':<15} | {'PSNR':<5} | {'JPG(KB)':<8} | {'WebP(KB)':<8} | {'Bits(KB)':<8} | {'Ratio(W/B)'}"
    print("\n" + "="*80)
    print(header)
    print("-" * 80)
    for _, r in df.iterrows():
        # This ratio is WebP size divided by Model Bitstream size
        post_flow_ratio = r['webp_kb'] / r['bitstream_kb']
        print(f"{r['file'][:15]:<15} | {r['psnr']:<5.1f} | {r['orig_jpg_kb']:<8.1f} | {r['webp_kb']:<8.1f} | {r['bitstream_kb']:<8.1f} | {post_flow_ratio:.2f}:1")
    
    print("="*80)
    # The final metric is the ratio of the WebP source to your Model output
    avg_ratio = df['webp_kb'].mean() / df['bitstream_kb'].mean()
    print(f"AVERAGE PSNR: {df['psnr'].mean():.2f} dB")
    print(f"AVERAGE RATIO (WebP Source / Model Bitstream): {avg_ratio:.2f}:1")

if __name__ == "__main__":
    run_webp_to_model_benchmark(epoch_num=15, max_images=10)
