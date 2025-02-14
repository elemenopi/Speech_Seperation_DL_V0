import os
import random
import shutil
from pathlib import Path

def move_random_folders(source_dir, destination_dir, num_of_folders):
    # Convert to Path objects
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    # Check if source and destination directories exist
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source directory '{source_dir}' does not exist or is not a directory.")
        return

    if not destination_dir.exists():
        print(f"Destination directory '{destination_dir}' does not exist. Creating it...")
        destination_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of all subdirectories in the source directory
    all_subdirs = [subdir for subdir in source_dir.iterdir() if subdir.is_dir()]

    # Check if num_of_folders is greater than the available subdirectories
    if num_of_folders > len(all_subdirs):
        print(f"Requested number of folders to move ({num_of_folders}) is greater than available ({len(all_subdirs)}).")
        num_of_folders = len(all_subdirs)

    # Randomly select folders to move
    folders_to_move = random.sample(all_subdirs, num_of_folders)

    # Move the selected folders
    for folder in folders_to_move:
        destination_path = destination_dir / folder.name
        print(f"Moving '{folder}' to '{destination_path}'")
        shutil.move(str(folder), str(destination_path))

    print(f"Moved {num_of_folders} folder(s) from '{source_dir}' to '{destination_dir}'.")

# Example usage
if __name__ == "__main__":
    source_directory = "/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds_ADVANCED"
    destination_directory = "/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Val"
    num_folders_to_move = 800

    move_random_folders(source_directory, destination_directory, num_folders_to_move)
