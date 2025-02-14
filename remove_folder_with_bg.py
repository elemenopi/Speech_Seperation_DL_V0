import os
import json
import shutil

# Define the source directory
source_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Val'  # Update this path accordingly

print(f"Scanning directory: {source_dir}")

# List to store folders to be deleted
folders_to_delete = []

# Iterate over each subfolder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        metadata_path = os.path.join(folder_path, 'metadata.json')
        # Check if the metadata.json file exists
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                # Check if any key contains 'bg'
                keys_with_bg = [key for key in data.keys() if 'bg' in key.lower()]
                if keys_with_bg:
                    folders_to_delete.append(folder_name)
            except json.JSONDecodeError as e:
                print(f"Error reading {metadata_path}: {e}")
                continue

# Report the number of folders to delete
num_folders = len(folders_to_delete)
print(f"Found {num_folders} folders with 'bg' in metadata keys in {source_dir}.")

if num_folders > 0:
    # List the folders (optional)
    print("The following folders will be deleted:")
    for folder_name in folders_to_delete:
        print(f"- {folder_name}")
    
    # Prompt for confirmation
    confirm = input("Do you want to proceed with deletion? (yes/no): ").strip().lower()
    if confirm == 'yes':
        # Delete the folders
        for folder_name in folders_to_delete:
            folder_path = os.path.join(source_dir, folder_name)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_name}")
            except Exception as e:
                print(f"Error deleting {folder_name}: {e}")
        print("Deletion completed.")
    else:
        print("Deletion canceled.")
else:
    print("No folders to delete.")
