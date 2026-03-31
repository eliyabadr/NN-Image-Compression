import torch
import torch.nn.functional as F
import math
import os
import time
import pandas as pd
import kagglehub
from PIL import Image
from torchvision.transforms import ToTensor
from train import SmallStudent # Ensure train.py is in the same directory

def run_dataset_benchmark(epoch_num=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Get the Kaggle Dataset Path
    print("Locating dataset...")
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Setup Model
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_imgs_60000.pth'
    model = SmallStudent(N=64, M=96).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)

    # 3. Collect all image paths
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(supported_ext):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to process.")

    results = []

    # 4. Benchmark Loop
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                # Load image
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                x = ToTensor()(img).unsqueeze(0).to(device)

                # Padding to 64-bit alignment
                p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
                x_padded = F.pad(x, p2d, "constant", 0)

                # Inference
                start_time = time.perf_counter()
                out = model(x_padded)
                elapsed = (time.perf_counter() - start_time) * 1000

                # Metrics Calculation
                x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
                mse = F.mse_loss(x, x_hat).item()
                psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100
                
                num_pixels = w * h
                total_bits = sum(
                    torch.log(likelihoods).sum().item() for likelihoods in out["likelihoods"].values()
                ) / -math.log(2)
                bpp = total_bits / num_pixels

                results.append({
                    "filename": os.path.basename(img_path),
                    "psnr": psnr,
                    "bpp": bpp,
                    "mse": mse,
                    "latency_ms": elapsed
                })

                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(image_paths)} images...")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # 5. Save and Summarize
    df_results = pd.DataFrame(results)
    output_csv = f"benchmark_results_epoch_{epoch_num}.csv"
    df_results.to_csv(output_csv, index=False)

    print("\n" + "="*30)
    print("BENCHMARK SUMMARY")
    print("="*30)
    print(f"Total Images:  {len(df_results)}")
    print(f"Average PSNR:  {df_results['psnr'].mean():.2f} dB")
    print(f"Average BPP:   {df_results['bpp'].mean():.4f}")
    print(f"Average MSE:   {df_results['mse'].mean():.6f}")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    run_dataset_benchmark(epoch_num=15)
