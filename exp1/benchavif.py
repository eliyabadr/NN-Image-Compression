import torch
import torch.nn.functional as F
import math
import os
import time
import pickle
import pillow_avif  # Vital for AVIF support in Pillow
from PIL import Image
from torchvision.transforms import ToTensor

# Assuming SmallStudent is defined in your training script
from train2_focus_on_compresion_ratio import SmallStudent 

def run_single_image_avif_benchmark(image_path, checkpoint_path):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Benchmarking on: {device}")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load Model
    # N and M should match your trained architecture (default N=64, M=96)
    model = SmallStudent(N=64, M=96).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)

    # 2. Process Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    orig_size_kb = os.path.getsize(image_path) / 1024

    # 3. Save as AVIF for comparison
    temp_avif = "comparison.avif"
    # quality=60 is a high-quality standard; speed=0 provides best compression
    img.save(temp_avif, "AVIF", quality=60, speed=0) 
    avif_size_kb = os.path.getsize(temp_avif) / 1024

    # 4. Model Compression
    x = ToTensor()(img).unsqueeze(0).to(device)
    
    # Pad to multiple of 64 (required by the architecture's 4 downsampling layers)
    pad_w = (64 - w % 64) % 64
    pad_h = (64 - h % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), "constant", 0)

    with torch.no_grad():
        # Compress
        start_comp = time.perf_counter()
        compressed_data = model.compress(x_padded) 
        comp_time = (time.perf_counter() - start_comp) * 1000

        # Measure bitstream size
        tmp_bin = "temp_bitstream.bin"
        with open(tmp_bin, "wb") as f:
            pickle.dump(compressed_data, f)
        bitstream_size_kb = os.path.getsize(tmp_bin) / 1024

        # Decompress
        start_decomp = time.perf_counter()
        out_decomp = model.decompress(compressed_data["strings"], compressed_data["shape"])
        decomp_time = (time.perf_counter() - start_decomp) * 1000
        
        # Crop padding and clamp
        x_hat = out_decomp["x_hat"][:, :, :h, :w].clamp(0, 1)
        
        # Calculate PSNR
        mse = F.mse_loss(x, x_hat).item()
        psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100

    # 5. Output Results
    print("\n" + "="*50)
    print(f"RESULTS FOR: {os.path.basename(image_path)}")
    print("-" * 50)
    print(f"Original Size:    {orig_size_kb:>10.2f} KB")
    print(f"AVIF Size:        {avif_size_kb:>10.2f} KB")
    print(f"Model Bitstream:  {bitstream_size_kb:>10.2f} KB")
    print("-" * 50)
    print(f"Model PSNR:       {psnr:>10.2f} dB")
    print(f"Encoding Time:    {comp_time:>10.2f} ms")
    print(f"Decoding Time:    {decomp_time:>10.2f} ms")
    print("-" * 50)
    print(f"Compression Ratio (Model vs Orig): {orig_size_kb/bitstream_size_kb:.2f}:1")
    print(f"Compression Ratio (AVIF vs Orig):  {orig_size_kb/avif_size_kb:.2f}:1")
    print("="*50)

    # Cleanup
    if os.path.exists(tmp_bin): os.remove(tmp_bin)
    # Keeping comparison.avif so you can view it if you want

if __name__ == "__main__":
    # Update these paths to your actual files
    MY_IMAGE = "path/to/your/image.jpg" 
    MY_CHECKPOINT = "/home/egb11/scratch/checkpoints/student_epoch_20_compression_focus.pth"
    
    run_single_image_avif_benchmark(MY_IMAGE, MY_CHECKPOINT)
