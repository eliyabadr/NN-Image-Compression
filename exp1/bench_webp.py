#:Wqon 2 (Optimized for Kaggle fetching)
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

    # 1. Locate Dataset (kagglehub handles caching automatically)
    print("Checking dataset...")
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model
    img_count = 60000 if epoch_num >= 10 else 40000
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_imgs_{img_count}.pth'
    
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

    print(f"Starting inference on {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # --- Original Disk Size ---
                orig_size_kb = os.path.getsize(img_path) / 1024

                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                x = ToTensor()(img).unsqueeze(0).to(device)

                # Padding to be divisible by 64
                p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
                x_padded = F.pad(x, p2d, "constant", 0)

                # 1. Compression
                start_comp = time.perf_counter()
                compressed_data = model.compress(x_padded) 
                comp_time = (time.perf_counter() - start_comp) * 1000

                # --- Bitstream Size ---
                tmp_bin = "temp_bitstream.bin"
                with open(tmp_bin, "wb") as f:
                    pickle.dump(compressed_data, f)
                compressed_size_kb = os.path.getsize(tmp_bin) / 1024

                # 2. Decompression
                start_decomp = time.perf_counter()
                out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
                decomp_time = (time.perf_counter() - start_decomp) * 1000
                
                # 3. Metrics
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
                    "bitstream_kb": compressed_size_kb
                })
            except Exception as e:
                print(f"Skipping {os.path.basename(img_path)} due to error: {e}")

    # 4. Display Results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print(f"{'Filename':<15} | {'PSNR':<6} | {'Enc(ms)':<8} | {'Dec(ms)':<8} | {'CompRatio'}")
        print("-" * 70)
        for _, r in df.iterrows():
            ratio = r['orig_kb'] / r['bitstream_kb'] if r['bitstream_kb'] > 0 else 0
            print(f"{r['file'][:15]:<15} | {r['psnr']:<6.2f} | {r['comp_ms']:<8.1f} | {r['decomp_ms']:<8.1f} | {ratio:.2f}:1")
        
        print("="*70)
        print(f"AVERAGE PSNR: {df['psnr'].mean():.2f} dB")
        print(f"AVERAGE COMPRESSION RATIO: {df['orig_kb'].mean() / df['bitstream_kb'].mean():.2f}:1")
    else:
        print("No images were successfully processed.")

    if os.path.exists("temp_bitstream.bin"): os.remove("temp_bitstream.bin")

if __name__ == "__main__":
    run_split_latency_benchmark(epoch_num=15, max_images=10)
