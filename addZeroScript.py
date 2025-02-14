import os

# Path to the directory containing the folders
base_dir = '/app/OUTPUTS_DSI/OUTPUTS_TRAIN/'

# Iterate through each folder in the directory
for folder_name in os.listdir(base_dir):
    # Check if the folder name matches the pattern 'outputNNNN'
    if folder_name.startswith('output') and len(folder_name) == 10:
        # Extract the numerical part of the folder name
        number_part = folder_name[6:]
        # Create the new folder name with leading zeros
        new_folder_name = f'output{int(number_part):05d}'
        # Construct the full path for the old and new folder names
        old_folder_path = os.path.join(base_dir, folder_name)
        new_folder_path = os.path.join(base_dir, new_folder_name)
        # Rename the folder
        os.rename(old_folder_path, new_folder_path)
        print(f'Renamed: {old_folder_path} to {new_folder_path}')

print("Renaming complete!")
