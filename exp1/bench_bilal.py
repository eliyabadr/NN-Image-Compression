import torch
import torch.nn.functional as F
import math
import os
import time
import kagglehub
from PIL import Image
from torchvision.transforms import ToTensor
from compressai.zoo import bmshj2018_hyperprior

def run_dataset_benchmark():
    device = 'cpu'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the BMSHJ Hyperprior model at highest quality (8)
    # Using pretrained=True pulls optimized weights from the CompressAI Zoo
    model = bmshj2018_hyperprior(quality=8, pretrained=True).to(device)
    model.eval()
    model.update(force=True)

    # 2. Get the Kaggle Dataset Path
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 3. Define the exact images mentioned in your output
    target_filenames = [
        "photo_24.jpg", "photo_23.jpg", "photo_4.jpg", "photo_3.jpg", 
        "portrait_1.jpg", "photo_18.jpg", "photo_16.jpg", "photo_11.jpg", 
        "photo_10.jpg", "photo_17.jpg"
    ]

    # Find the full paths for these specific files within the dataset
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file in target_filenames:
                image_paths.append(os.path.join(root, file))
    
    # Sort them to match your list order
    image_paths.sort(key=lambda x: target_filenames.index(os.path.basename(x)))

    print(f"\n{'='*51}")
    print(f"{'Filename':<20} | {'PSNR':<7} | {'Comp (ms)':<10} | {'Decomp (ms)':<10}")
    print(f"{'-'*51}")

    total_comp = 0
    total_decomp = 0

    # 4. Benchmark Loop
    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                x = ToTensor()(img).unsqueeze(0).to(device)

                # Padding to 64-bit alignment (Required for BMHSJ architecture)
                p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
                x_padded = F.pad(x, p2d, "constant", 0)

                # Measure Compression (Encoding)
                start_comp = time.perf_counter()
                out_enc = model.compress(x_padded)
                comp_time = (time.perf_counter() - start_comp) * 1000

                # Measure Decompression (Decoding)
                start_decomp = time.perf_counter()
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                decomp_time = (time.perf_counter() - start_decomp) * 1000

                # PSNR Calculation
                x_hat = out_dec["x_hat"][:, :, :h, :w].clamp(0, 1)
                mse = F.mse_loss(x, x_hat).item()
                psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100

                fname = os.path.basename(img_path)
                print(f"{fname:<20} | {psnr:<7.2f} | {comp_time:<10.2f} | {decomp_time:<10.2f}")
                
                total_comp += comp_time
                total_decomp += decomp_time

            except Exception as e:
                print(f"Error processing {os.path.basename(img_path)}: {e}")

    # 5. Final Summary
    avg_comp = total_comp / len(image_paths) if image_paths else 0
    avg_decomp = total_decomp / len(image_paths) if image_paths else 0

    print(f"{'='*51}")
    print(f"AVERAGE COMPRESSION:   {avg_comp:.2f} ms")
    print(f"AVERAGE DECOMPRESSION: {avg_decomp:.2f} ms")

if __name__ == "__main__":
    run_dataset_benchmark()
