import os
import json
import shutil

# Define the source directory
source_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Val'

# List to store folders to be deleted
folders_to_delete = []

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
                    folders_to_delete.append(folder_name)

# Report the number of folders to delete
num_folders = len(folders_to_delete)
print(f"Found {num_folders} folders with exactly five speakers.")

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
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_name}")
        print("Deletion completed.")
    else:
        print("Deletion canceled.")
else:
    print("No folders to delete.")
