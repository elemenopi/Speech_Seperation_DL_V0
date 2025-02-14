import os
import json
import shutil

def validate_metadata_file(metadata_path):
    try:
        with open(metadata_path, 'r') as json_file:
            metadata = json.load(json_file)

        # Check for voice entries
        for key, value in metadata.items():
            if key.startswith('voice'):
                if 'Position' not in value or 'speaker_id' not in value:
                    print(f"Invalid 'voice' entry in {metadata_path}: Missing 'Position' or 'speaker_id'")
                    return False
                if not isinstance(value['Position'], list) or len(value['Position']) != 3:
                    print(f"Invalid 'Position' format in {metadata_path} for {key}")
                    return False

            # Check for background entries (optional)
            elif key.startswith('bg'):
                if 'position' not in value:
                    print(f"Invalid 'bg' entry in {metadata_path}: Missing 'position'")
                    return False
                if not isinstance(value['position'], list) or len(value['position']) != 3:
                    print(f"Invalid 'position' format in {metadata_path} for {key}")
                    return False
        return True

    except json.JSONDecodeError:
        print(f"JSON decoding error in {metadata_path}")
        return False
    except Exception as e:
        print(f"Error processing {metadata_path}: {e}")
        return False

def find_invalid_metadata(directory):
    invalid_folders = []

    # List all items in the given directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        
        # Check if the item is a directory and starts with 'output'
        if os.path.isdir(folder_path) and folder.startswith('output'):
            # Check if 'metadata.json' exists in this directory
            metadata_path = os.path.join(folder_path, 'metadata.json')
            if os.path.isfile(metadata_path):
                if not validate_metadata_file(metadata_path):
                    invalid_folders.append(folder_path)
            else:
                print(f"No 'metadata.json' found in {folder_path}")
                invalid_folders.append(folder_path)

    return invalid_folders

def remove_folders(folders):
    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"Removed {folder}")
        except Exception as e:
            print(f"Error removing {folder}: {e}")

def main():
    directory = 'gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Train_simple_3d_0610'  # Replace with your directory
    invalid_folders = find_invalid_metadata(directory)
    
    print(f"Total folders with invalid 'metadata.json': {len(invalid_folders)}")
    for folder in invalid_folders:
        print(folder)

    # Confirm before removing
    confirm = input("Do you want to delete these folders? Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        remove_folders(invalid_folders)
    else:
        print("No folders were removed.")

if __name__ == "__main__":
    main()
