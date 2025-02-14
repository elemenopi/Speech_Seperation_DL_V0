import os

def find_folders_without_metadata(directory):
    folders_without_metadata = []

    # List all items in the given directory
    for folder in os.listdir(directory):
        print(folder)
        folder_path = os.path.join(directory, folder)
        
        # Check if the item is a directory and starts with 'output'
        if os.path.isdir(folder_path) and folder.startswith('output'):
            # Check if 'metadata.json' exists in this directory
            metadata_path = os.path.join(folder_path, 'metadata.json')
            if not os.path.isfile(metadata_path):
                folders_without_metadata.append(folder_path)

    return folders_without_metadata

def main():
    directory = 'OUTPUTS_DSI/OUTPUTS_TRAIN'  # Replace with your directory
    folders = find_folders_without_metadata(directory)
    
    print(f"Total folders without 'metadata.json': {len(folders)}")
    for folder in folders:
        print(folder)

if __name__ == "__main__":
    main()