import shutil
from pathlib import Path

def copy_folders_by_pattern(source_dir, destination_dir, pattern):
    # Convert to Path objects
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    # Check if source directory exists
    if not source_dir.exists() or not source_dir.is_dir():
        #print(f"Source directory '{source_dir}' does not exist or is not a directory.")
        return

    # Create the destination directory if it doesn't exist
    if not destination_dir.exists():
        #print(f"Destination directory '{destination_dir}' does not exist. Creating it...")
        destination_dir.mkdir(parents=True, exist_ok=True)

    # Find all folders in the source directory matching the pattern
    matching_folders = [subdir for subdir in source_dir.iterdir() if subdir.is_dir() and subdir.name.startswith(pattern)]

    # Copy the matching folders to the destination
    for folder in matching_folders:
        destination_path = destination_dir / folder.name
        #print(f"Copying '{folder}' to '{destination_path}'")
        shutil.copytree(folder, destination_path)

    #print(f"Copied {len(matching_folders)} folder(s) from '{source_dir}' to '{destination_dir}'.")

# Example usage
if __name__ == "__main__":
    source_directory = "/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds_ADVANCED"
    destination_directory = "/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/outputDGX03"
    folder_pattern = "output"  # This will match all folders starting with 'output20'

    copy_folders_by_pattern(source_directory, destination_directory, folder_pattern)
