import torch
import torch.nn.functional as F
import math
import os
import time
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# Import the model architecture defined in your uploaded file
from train import SmallStudent 

def run_inference_and_stats(epoch_num=9):
    device = 'cpu'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Paths
    # Using the exact filename you found in your directory
    checkpoint_path = f'/home/egb11/scratch/exp1/checkpoints/student_epoch_{epoch_num}_imgs_20000.pth'
    # Update this to an existing image in your scratch directory
    img_path = '/home/egb11/scratch/my_images/60000/60500.png' 
    output_img_path = f'reconstructed_epoch_{epoch_num}.png'

    # 2. Initialize and Load Model
    # N=64, M=96 matches your SmallStudent definition
    model = SmallStudent(N=64, M=96).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Critical Fix: Load only the model weights from the dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True) # Build entropy tables for BPP calculation

    # 3. Process Image
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x = ToTensor()(img).unsqueeze(0).to(device)

    # Padding to 64-bit alignment (Required for CompressAI models)
    p2d = (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64)
    x_padded = F.pad(x, p2d, "constant", 0)

    # 4. Run Inference
    with torch.no_grad():
        start_time = time.perf_counter()
        out = model(x_padded)
        end_time = time.perf_counter()

    # 5. Reconstruction
    # Crop the padded version back to original size
    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
    reconstructed_img = ToPILImage()(x_hat.squeeze(0).cpu())
    reconstructed_img.save(output_img_path)

    # 6. Calculate Statistics (as defined in fyp_1(1).py)
    mse = F.mse_loss(x, x_hat).item()
    # PSNR = 20 * log10(1 / sqrt(MSE))
    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100
    
    num_pixels = w * h
    # BPP = sum(-log2(likelihoods)) / num_pixels
    total_bits = sum(
        torch.log(likelihoods).sum().item() for likelihoods in out["likelihoods"].values()
    ) / -math.log(2)
    bpp = total_bits / num_pixels

    # 7. Print Resultsssuming a standard 24-bit RGB image
    original_bpp = 24 
    compression_ratio = original_bpp / bpp

# Calculating space saved as a percentage
    space_saved = (1 - (bpp / original_bpp)) * 100

    print(f"Compression Ratio:   {compression_ratio:.2f}:1")
    print(f"Space Saved:         {space_saved:.2f}%")
    print(f"\n{'--- Inference Results ---':^40}")
    print(f"Reconstructed Image: {output_img_path}")
    print(f"PSNR (Quality):      {psnr:.2f} dB")
    print(f"BPP (Rate):          {bpp:.4f}")
    print(f"MSE:                 {mse:.6f}")
    print(f"Latency:             {(end_time - start_time)*1000:.2f} ms")
    print(f"image compression ratio : {24/bpp:.2f}:1")
    print(f"Model Disk Size : {os.path.getsize(checkpoint_path) / 1e6:.2f} MB")
if __name__ == "__main__":
    # Test with epoch 1 (your second epoch) as it has more training
    run_inference_and_stats(epoch_num=1)
