import math
import pickle
import shutil
import time
import zlib
from datetime import datetime
from pathlib import Path

import kagglehub
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from train import SmallStudent

try:
    from pytorch_msssim import ms_ssim, ssim
except Exception:
    ms_ssim = None
    ssim = None


# ============================================================
# Fixed configuration
# ============================================================
CHECKPOINT_PATH = Path(
    "/home/egb11/scratch/exp1/checkpoints/student_epoch_20_imgs_60000.pth"
)
DATASET_ID = "osmankagankurnaz/human-profile-photos-dataset"
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ARCHIVE_FORMAT = "neurarchive-v3-metrics-mb"
PAD_MULTIPLE = 64
OUTPUT_BASE_DIR = Path("/home/egb11/scratch/exp1")
RUNS_DIR_NAME = "compression_runs"
DEFAULT_MAX_IMAGES = 25


# ============================================================
# Model loading
# ============================================================
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {CHECKPOINT_PATH}")
    print(f"Benchmarking on: {device}")

    model = SmallStudent(N=64, M=96).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    if hasattr(model, "update"):
        model.update(force=True)

    return model, device


# ============================================================
# Helpers
# ============================================================
def bytes_to_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def pad_to_multiple(x, multiple=PAD_MULTIPLE):
    _, _, h, w = x.shape
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return padded, h, w


def list_images(input_folder):
    input_folder = Path(input_folder)
    return sorted(
        p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def num_pixels_from_hw(h, w):
    return int(h) * int(w)


def compute_psnr(x, y, data_range=1.0):
    mse = F.mse_loss(x, y).item()
    if mse <= 1e-12:
        return float("inf"), 0.0
    psnr = 10.0 * math.log10((data_range ** 2) / mse)
    return psnr, mse


def compute_mae(x, y):
    return torch.mean(torch.abs(x - y)).item()


def compute_rmse(mse):
    return math.sqrt(max(mse, 0.0))


def compute_optional_ssim_metrics(x, y):
    if ssim is None or ms_ssim is None:
        return None, None

    _, _, h, w = x.shape
    min_side = min(h, w)
    ssim_value = float(ssim(x, y, data_range=1.0).item())
    ms_ssim_value = None if min_side < 161 else float(ms_ssim(x, y, data_range=1.0).item())
    return ssim_value, ms_ssim_value


def bits_per_pixel(num_bytes, h, w):
    pixels = num_pixels_from_hw(h, w)
    if pixels <= 0:
        return 0.0
    return (8.0 * float(num_bytes)) / float(pixels)


def get_model_bitstream_bytes(strings_obj):
    if isinstance(strings_obj, (list, tuple)):
        total = 0
        for group in strings_obj:
            if isinstance(group, (list, tuple)):
                total += sum(len(s) for s in group)
            else:
                total += len(group)
        return int(total)
    return int(len(strings_obj))


def create_run_dirs(base_dir=OUTPUT_BASE_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(base_dir) / RUNS_DIR_NAME / f"run_{timestamp}"
    subset_dir = run_root / "input_subset"
    restored_dir = run_root / "restored_images"
    archive_path = run_root / "images_bundle.narc"
    csv_path = run_root / "benchmark_metrics.csv"

    run_root.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)
    return run_root, subset_dir, restored_dir, archive_path, csv_path


# ============================================================
# Per-image compression / decompression
# ============================================================
@torch.no_grad()
def compress_single_image(model, device, img_path):
    img_path = Path(img_path)
    img = Image.open(img_path).convert("RGB")
    x = ToTensor()(img).unsqueeze(0).to(device)
    x_padded, orig_h, orig_w = pad_to_multiple(x, multiple=PAD_MULTIPLE)

    start_comp = time.perf_counter()
    compressed = model.compress(x_padded)
    comp_ms = (time.perf_counter() - start_comp) * 1000.0

    record = {
        "name": img_path.name,
        "orig_size": (orig_h, orig_w),
        "shape": compressed["shape"],
        "strings": compressed["strings"],
    }
    return record, comp_ms, x[:, :, :orig_h, :orig_w].detach().cpu()


@torch.no_grad()
def decompress_single_image_tensor(model, device, record):
    start_decomp = time.perf_counter()
    out = model.decompress(record["strings"], record["shape"])
    decomp_ms = (time.perf_counter() - start_decomp) * 1000.0

    h, w = record["orig_size"]
    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
    return x_hat, decomp_ms


@torch.no_grad()
def decompress_single_image(model, device, record):
    x_hat, decomp_ms = decompress_single_image_tensor(model, device, record)
    img = ToPILImage()(x_hat.cpu().squeeze(0))
    return img, decomp_ms


# ============================================================
# Archive compression with descriptive metrics
# ============================================================
@torch.no_grad()
def compress_folder_to_archive(input_folder, output_archive, model, device, use_outer_zlib=True):
    input_folder = Path(input_folder)
    output_archive = Path(output_archive)

    image_paths = list_images(input_folder)
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_folder}")

    archive_records = []
    stats = []

    for img_path in image_paths:
        record, comp_ms, x_orig_cpu = compress_single_image(model, device, img_path)
        rel_name = str(img_path.relative_to(input_folder))
        record["name"] = rel_name

        model_bytes = get_model_bitstream_bytes(record["strings"])
        x_hat, decomp_ms = decompress_single_image_tensor(model, device, record)
        x_hat_cpu = x_hat.detach().cpu()

        h, w = record["orig_size"]
        pixels = num_pixels_from_hw(h, w)
        orig_disk_bytes = img_path.stat().st_size
        raw_record_bytes = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        packed_record_bytes = len(raw_record_bytes)

        psnr, mse = compute_psnr(x_orig_cpu, x_hat_cpu, data_range=1.0)
        mae = compute_mae(x_orig_cpu, x_hat_cpu)
        rmse = compute_rmse(mse)
        ssim_value, ms_ssim_value = compute_optional_ssim_metrics(x_orig_cpu, x_hat_cpu)

        archive_records.append(record)
        stats.append(
            {
                "file": rel_name,
                "height": h,
                "width": w,
                "pixels": pixels,
                "orig_disk_mb": bytes_to_mb(orig_disk_bytes),
                "model_bitstream_mb": bytes_to_mb(model_bytes),
                "packed_record_mb": bytes_to_mb(packed_record_bytes),
                "bpp_model": bits_per_pixel(model_bytes, h, w),
                "bpp_record": bits_per_pixel(packed_record_bytes, h, w),
                "ratio_disk_to_model": orig_disk_bytes / max(model_bytes, 1),
                "ratio_disk_to_record": orig_disk_bytes / max(packed_record_bytes, 1),
                "space_saving_pct_model": 100.0 * (1.0 - (model_bytes / max(orig_disk_bytes, 1))),
                "psnr_db": psnr,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "ssim": ssim_value,
                "ms_ssim": ms_ssim_value,
                "encode_ms": comp_ms,
                "decode_ms_inline": decomp_ms,
            }
        )

    archive_obj = {
        "format": ARCHIVE_FORMAT,
        "count": len(archive_records),
        "records": archive_records,
    }

    payload = pickle.dumps(archive_obj, protocol=pickle.HIGHEST_PROTOCOL)
    if use_outer_zlib:
        payload = zlib.compress(payload, level=9)

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with open(output_archive, "wb") as f:
        f.write(payload)

    df = pd.DataFrame(stats).sort_values("file").reset_index(drop=True)
    total_orig_mb = df["orig_disk_mb"].sum()
    total_model_mb = df["model_bitstream_mb"].sum()
    archive_mb = bytes_to_mb(output_archive.stat().st_size)

    print(f"Saved archive: {output_archive}")
    print(f"Images packed: {len(image_paths)}")
    print(f"Total original disk size: {total_orig_mb:.4f} MB")
    print(f"Total model bitstream size: {total_model_mb:.4f} MB")
    print(f"Final archive size on disk: {archive_mb:.4f} MB")
    print(f"Model-only ratio (disk/model): {total_orig_mb / max(total_model_mb, 1e-12):.2f}:1")
    print(f"Archive ratio (disk/archive): {total_orig_mb / max(archive_mb, 1e-12):.2f}:1")
    print(f"Average encode time: {df['encode_ms'].mean():.2f} ms/image")
    print(f"Average decode time: {df['decode_ms_inline'].mean():.2f} ms/image")
    print(f"Average PSNR: {df['psnr_db'].mean():.3f} dB")
    print(f"Average BPP (model): {df['bpp_model'].mean():.4f}")
    if df["ssim"].notna().any():
        print(f"Average SSIM: {df['ssim'].dropna().mean():.6f}")
    if df["ms_ssim"].notna().any():
        print(f"Average MS-SSIM: {df['ms_ssim'].dropna().mean():.6f}")

    return df, archive_mb


@torch.no_grad()
def decompress_archive_to_folder(archive_path, output_folder, model, device):
    archive_path = Path(archive_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "rb") as f:
        payload = f.read()

    try:
        archive_obj = pickle.loads(zlib.decompress(payload))
    except zlib.error:
        archive_obj = pickle.loads(payload)

    if archive_obj.get("format") != ARCHIVE_FORMAT:
        raise ValueError(f"Unsupported archive format: {archive_obj.get('format')}")

    stats = []
    for record in archive_obj["records"]:
        img, decomp_ms = decompress_single_image(model, device, record)
        out_path = output_folder / record["name"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            img.save(out_path, quality=95)
        else:
            img.save(out_path)

        stats.append(
            {
                "file": record["name"],
                "decode_ms_save_pass": decomp_ms,
                "restored_disk_mb": bytes_to_mb(out_path.stat().st_size),
                "saved_path": str(out_path),
            }
        )

    df = pd.DataFrame(stats).sort_values("file").reset_index(drop=True)
    print(f"Restored {len(df)} images into: {output_folder}")
    print(f"Average decode time (restore pass): {df['decode_ms_save_pass'].mean():.2f} ms/image")
    return df


# ============================================================
# Kaggle benchmark helper
# ============================================================
def prepare_kaggle_subset(max_images, subset_dir):
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    all_files = list_images(dataset_path)
    if not all_files:
        raise ValueError(f"No images found in downloaded dataset: {dataset_path}")

    subset_dir = Path(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    selected = sorted(all_files)[:max_images]
    copied = []

    for src in selected:
        dst = subset_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    return dataset_path, subset_dir, copied


# ============================================================
# High-level benchmark runner
# ============================================================
@torch.no_grad()
def run_folder_archive_benchmark(max_images=DEFAULT_MAX_IMAGES, csv_path=None):
    model, device = load_model()

    run_root, subset_dir, restored_dir, archive_path, default_csv_path = create_run_dirs()
    dataset_path, bench_root, copied = prepare_kaggle_subset(max_images=max_images, subset_dir=subset_dir)

    print(f"Using dataset cache: {dataset_path}")
    print(f"Run output folder: {run_root}")
    print(f"Input subset folder: {bench_root}")
    print(f"Restored images folder: {restored_dir}")
    print(f"Selected images: {len(copied)}")

    metrics_df, archive_mb = compress_folder_to_archive(
        input_folder=bench_root,
        output_archive=archive_path,
        model=model,
        device=device,
        use_outer_zlib=True,
    )

    restore_df = decompress_archive_to_folder(
        archive_path=archive_path,
        output_folder=restored_dir,
        model=model,
        device=device,
    )

    merged_df = metrics_df.merge(restore_df, on="file", how="left")

    summary_cols = [
        "file",
        "height",
        "width",
        "orig_disk_mb",
        "model_bitstream_mb",
        "restored_disk_mb",
        "bpp_model",
        "ratio_disk_to_model",
        "space_saving_pct_model",
        "psnr_db",
        "mse",
        "rmse",
        "mae",
        "ssim",
        "ms_ssim",
        "encode_ms",
        "decode_ms_inline",
        "decode_ms_save_pass",
        "saved_path",
    ]

    printable_df = merged_df[summary_cols].copy()

    print("\nPer-image detailed summary (MB-based):")
    print(printable_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if csv_path is None:
        csv_path = default_csv_path
    csv_path = Path(csv_path)
    printable_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-image metrics CSV to: {csv_path}")
    print(f"Saved reconstructed images to: {restored_dir}")
    print(f"Saved archive to: {archive_path}")

    return printable_df, archive_mb, run_root


if __name__ == "__main__":
    run_folder_archive_benchmark(max_images=DEFAULT_MAX_IMAGES)
