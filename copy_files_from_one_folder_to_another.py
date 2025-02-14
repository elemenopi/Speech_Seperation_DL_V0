import os
import shutil

# Source and destination directories
src_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/outputDGX03'
dest_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds_ADVANCED'

# Ensure destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Loop through all items in the source directory
for item in os.listdir(src_dir):
    src_path = os.path.join(src_dir, item)
    dest_path = os.path.join(dest_dir, item)

    # Copy only directories
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest_path)

print("Copying completed.")
