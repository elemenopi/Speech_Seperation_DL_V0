import os
import shutil

def get_latest_number(folder, prefix):
    existing_folders = [folder_name for folder_name in os.listdir(folder) if os.path.isdir(os.path.join(folder, folder_name)) and folder_name.startswith(prefix)]
    if not existing_folders:
        return -1
    latest_folder = max(existing_folders, key=lambda x: int(x[len(prefix):]))
    return int(latest_folder[len(prefix):])

def copy_and_rename_folders(src_folder, dst_folder, prefix):
    # Get the latest number in the destination folder
    latest_number = get_latest_number(dst_folder, prefix)
    
    # Get a list of folders in the source folder
    src_folders = [folder_name for folder_name in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, folder_name)) and folder_name.startswith(prefix)]
    
    for folder_name in sorted(src_folders):
        latest_number += 1
        new_folder_name = f"{prefix}{latest_number:04d}"
        src_path = os.path.join(src_folder, folder_name)
        dst_path = os.path.join(dst_folder, new_folder_name)
        
        shutil.copytree(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")
        
        shutil.rmtree(src_path)
        print(f"Deleted {src_path}")

# Usage example:
src_folder = "."
dst_folder = "/dsi/gannot-lab1/projects_2024/Ilya_Tomer/OUTPUTS/OUTPUTS_TRAIN"
prefix = "output"

copy_and_rename_folders(src_folder, dst_folder, prefix)
