import os

def rename_folders(base_folder, folder_prefix):
    # Get a list of existing folders in the base directory
    existing_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder)) and folder.startswith(folder_prefix)]
    
    # Sort folders by their numeric value
    existing_folders.sort(key=lambda x: int(x[len(folder_prefix):]))

    for folder in existing_folders:
        folder_number = int(folder[len(folder_prefix):])
        new_folder_name = f"{folder_prefix}{folder_number:04d}"
        
        old_folder_path = os.path.join(base_folder, folder)
        new_folder_path = os.path.join(base_folder, new_folder_name)
        
        # Rename the folder
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed {old_folder_path} to {new_folder_path}")

# Usage example:
base_folder = "."
folder_prefix = "output"
rename_folders(base_folder, folder_prefix)