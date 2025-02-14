import os
import shutil

def find_folders_without_metadata(directory):
    folders_without_metadata = []

    # List all items in the given directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        
        # Check if the item is a directory and starts with 'output'
        if os.path.isdir(folder_path) and folder.startswith('output'):
            # Check if 'metadata.json' exists in this directory
            metadata_path = os.path.join(folder_path, 'metadata.json')
            if not os.path.isfile(metadata_path):
                folders_without_metadata.append(folder_path)

    return folders_without_metadata

def remove_folders(folders):
    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"Removed {folder}")
        except Exception as e:
            print(f"Error removing {folder}: {e}")

def main():
    directory = 'gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/eval_sisdr_2410_bg'  # Replace with your directory
    folders = find_folders_without_metadata(directory)
    for folder in folders:
        print(folder)
    print(f"Total folders without 'metadata.json': {len(folders)}")

    confirm = input("Do you want to delete these folders? Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        remove_folders(folders)
        
    else:
        print("No folders were removed.")

if __name__ == "__main__":
    main()