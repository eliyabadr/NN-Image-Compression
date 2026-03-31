import os
import kagglehub

def check_image_disk_sizes(max_images=10):
    # 1. Locate the dataset
    print("Locating dataset...")
    dataset_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")
    
    # 2. Identify image files
    supported_ext = ('.jpg', '.jpeg', '.png')
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(supported_ext):
                image_paths.append(os.path.join(root, f))
    
    # Limit to 10 images as requested
    targets = image_paths[:max_images]
    
    print(f"\n{'Filename':<30} | {'Size (KB)':<12}")
    print("-" * 45)
    
    total_kb = 0
    for img_path in targets:
        # Get size in bytes and convert to KB
        size_bytes = os.path.getsize(img_path)
        size_kb = size_bytes / 1024
        total_kb += size_kb
        
        filename = os.path.basename(img_path)
        print(f"{filename[:30]:<30} | {size_kb:<12.2f}")
    
    print("-" * 45)
    print(f"Total Disk Size (10 images): {total_kb:.2f} KB ({total_kb/1024:.2f} MB)")

if __name__ == "__main__":
    check_image_disk_sizes(max_images=10)
