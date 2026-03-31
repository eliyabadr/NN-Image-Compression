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

# --- 2. HPC DATASET HANDLER ---
class HPCDataset(Dataset):
    def __init__(self, root_dir, max_images, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Get sorted list of subfolders (00000, 01000, etc.)
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for f in files:
                self.image_paths.append(os.path.join(subdir_path, f))
                if len(self.image_paths) >= max_images:
                    break
            if len(self.image_paths) >= max_images:
                break
        
        print(f"Dataset initialized with {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --- 3. DISTILLATION TRAINER ---
class DistilledTrainer(nn.Module):
    def __init__(self, student, teacher_quality=3, lmbda=0.01, distill_weight=0.5):
        super().__init__()
        self.student = student
        self.teacher = bmshj2018_hyperprior(quality=teacher_quality, pretrained=True).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.lmbda = lmbda
        self.distill_weight = distill_weight
        self.adapter = nn.Conv2d(student.M, 192, kernel_size=1)

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
        distill_loss = F.mse_loss(self.adapter(y_student), y_teacher)

        total_loss = bpp_loss + self.lmbda * (mse_dist * 255**2) + self.distill_weight * distill_loss
        return total_loss, bpp_loss, mse_dist

# --- 4. MAIN TRAINING LOOP ---
def run_hpc_training(max_imgs=5000, epochs=10, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data Setup
    transform = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor()])
    dataset = HPCDataset(root_dir='/home/egb11/scratch/my_images', max_images=max_imgs, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model Setup
    student = SmallStudent().to(device)
    trainer = DistilledTrainer(student).to(device)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    aux_optimizer = torch.optim.Adam(student.entropy_bottleneck.parameters(), lr=1e-3)

    for epoch in range(epochs):
        student.train()
        for i, images in enumerate(loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loss, bpp, mse = trainer(images)
            loss.backward()
            optimizer.step()

            aux_optimizer.zero_grad()
            aux_loss = student.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] | Loss: {loss.item():.4f} | BPP: {bpp.item():.3f}")

        # --- SAVE CHECKPOINT AT EACH EPOCH ---
        student.update(force=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"student_epoch_{epoch}_imgs_{max_imgs}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    # Specify the number of images to pull from the beginning of the dataset
    run_hpc_training(max_imgs=20000, epochs=20)
