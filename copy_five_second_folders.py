import os
import json
import shutil

# Define the source and target directories
source_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds_ADVANCED'
target_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/fivespeakers'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate over each subfolder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        metadata_path = os.path.join(folder_path, 'metadata.json')
        # Check if the metadata.json file exists
        if os.path.isfile(metadata_path):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                # Count the number of keys that start with 'voice'
                voice_keys = [key for key in data.keys() if key.startswith('voice')]
                if len(voice_keys) == 5:
                    # Copy the entire folder to the target directory
                    dest_path = os.path.join(target_dir, folder_name)
                    shutil.copytree(folder_path, dest_path)
                    print(f"Copied folder {folder_name} to {dest_path}")
