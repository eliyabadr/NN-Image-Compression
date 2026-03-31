#Updated Version 1
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from compressai.zoo import bmshj2018_hyperprior
from compressai.models.google import ScaleHyperprior
from compressai.layers.layers import GDN
from compressai.models.utils import conv, deconv

# --- 1. MODEL DEFINITION ---
class SmallStudent(ScaleHyperprior):
    def __init__(self, N=64, M=96):
        super().__init__(N=N, M=M)
        self.M = M
        self.g_a = nn.Sequential(
            conv(3, N, stride=2), GDN(N),
            conv(N, N, stride=2), GDN(N),
            conv(N, N, stride=2), GDN(N),
            conv(N, M, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, stride=2), GDN(N, inverse=True),
            deconv(N, N, stride=2), GDN(N, inverse=True),
            deconv(N, N, stride=2), GDN(N, inverse=True),
            deconv(N, 3, stride=2),
        )

# --- 2. DATASET ---
class HPCDatasetOffset(Dataset):
    def __init__(self, root_dir, max_images, offset, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        all_paths = []
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files:
                all_paths.append(os.path.join(subdir_path, f))
        self.image_paths = all_paths[offset : offset + max_images]
        print(f"Compression Focus: Loaded {len(self.image_paths)} images (Offset: {offset})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --- 3. COMPRESSION-FOCUSED TRAINER ---
class DistilledTrainer(nn.Module):
    def __init__(self, student, lmbda=0.001): # Lower lambda = Higher compression
        super().__init__()
        self.student = student
        self.teacher = bmshj2018_hyperprior(quality=6, pretrained=True).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.lmbda = lmbda
        teacher_channels = self.teacher.g_a[-1].out_channels
        self.adapter = nn.Conv2d(student.M, teacher_channels, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            y_teacher = self.teacher.g_a(x)

        y_student = self.student.g_a(x)
        z_student = self.student.h_a(y_student)
        z_hat, z_likelihoods = self.student.entropy_bottleneck(z_student)
        scales_hat = self.student.h_s(z_hat)
        y_hat, y_likelihoods = self.student.gaussian_conditional(y_student, scales_hat)
        x_hat = self.student.g_s(y_hat)

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp_loss = sum(torch.log(lik).sum() for lik in [y_likelihoods, z_likelihoods]) / (-math.log(2) * num_pixels)
        mse_dist = F.mse_loss(x_hat, x)
        
        # Reduced distillation weight (0.1) to favor compression autonomy
        distill_loss = F.mse_loss(self.adapter(y_student), y_teacher)

        # TOTAL LOSS: Focus is on bpp_loss (compression ratio)
        total_loss = bpp_loss + self.lmbda * (mse_dist * 255**2) + 0.1 * distill_loss
        return total_loss, bpp_loss, mse_dist

# --- 4. EXECUTION ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "./checkpoints"
    
    transform = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor()])
    # Loading 10,000 images starting after the previous 60k block
    dataset = HPCDatasetOffset(root_dir='/home/egb11/scratch/my_images', 
                               max_images=10000, offset=20000, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    student = SmallStudent(N=64, M=96).to(device)
    # λ = 0.001 (Extremely aggressive compression focus)
    trainer = DistilledTrainer(student, lmbda=0.001).to(device)
    
    optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4)
    aux_optimizer = torch.optim.Adam(student.entropy_bottleneck.parameters(), lr=1e-3)

    # Load Epoch 15 Checkpoint
    load_path = os.path.join(checkpoint_dir, "student_epoch_15_imgs_60000.pth")
    if os.path.exists(load_path):
        print(f"Resuming from Epoch 19: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Error: Could not find epoch 15 checkpoint.")
        return

    for epoch in range(16, 25): 
        student.train()
        print(f"\n--- Epoch {epoch} | Focus: Compression Ratio (BPP) ---")
        for i, images in enumerate(loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loss, bpp, mse = trainer(images)
            loss.backward()
            optimizer.step()

            aux_optimizer.zero_grad()
            student.aux_loss().backward()
            aux_optimizer.step()

            if i % 20 == 0:
                print(f"Step [{i}/{len(loader)}] | Loss: {loss.item():.4f} | BPP: {bpp.item():.4f} | MSE: {mse.item():.5f}")

        student.update(force=True)
        save_path = os.path.join(checkpoint_dir, f"student_epoch_{epoch}_sub_route_compression_focus.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_training()
