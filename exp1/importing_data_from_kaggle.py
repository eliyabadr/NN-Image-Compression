import kagglehub
import os

# 1. Download the dataset files
# This returns the local path where the images were saved
local_path = kagglehub.dataset_download("trainingdatapro/portrait-and-30-photos-test")

print(f"Dataset downloaded to: {local_path}")

# 2. List the files to see the images
files = os.listdir(local_path)
print(f"Found {len(files)} items in the directory.")
for f in files[:5]:  # Print the first 5 filenames
    print(f"- {f}")
