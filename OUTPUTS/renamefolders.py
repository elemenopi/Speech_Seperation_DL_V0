import os

def rename_folders(folder, prefix, start_number):
    # Get a list of folders in the directory that match the prefix
    existing_folders = [folder_name for folder_name in os.listdir(folder) if os.path.isdir(os.path.join(folder, folder_name)) and folder_name.startswith(prefix)]
    
    # Sort folders by their numeric value
    existing_folders.sort(key=lambda x: int(x[len(prefix):]))
    
    # Rename folders to a continuous sequence starting from start_number
    current_number = start_number
    
    for folder_name in existing_folders:
        new_folder_name = f"{prefix}{current_number:04d}"
        src_path = os.path.join(folder, folder_name)
        dst_path = os.path.join(folder, new_folder_name)
        
        os.rename(src_path, dst_path)
        print(f"Renamed {src_path} to {dst_path}")
        
        current_number += 1

# Usage example:
folder = "."
prefix = "output"
start_number = 3143  # Starting number for renaming

rename_folders(folder, prefix, start_number)
