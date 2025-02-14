import os

# Starting value for the renaming
new_index = 5549

# Directory containing the folders
base_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/outputDSIGPU04'  # Change this to the path of your folders if needed

#DGX03
#DSIGPU02 16704 done
#DSIGPU05 18445     done     total : 3914
#DSIGPU03 22358 total : 3690 done
basis = 0
# Loop through numbers from 00000 to 02500
for i in range(3946):
    # Create the old folder name with zero padding
    old_folder_name = f"output{(i+basis):05d}"
    old_folder_path = os.path.join(base_dir, old_folder_name)
    
    # Check if the folder exists
    if os.path.isdir(old_folder_path):
        # Create the new folder name
        new_folder_name = f"output{new_index:05d}"
        new_folder_path = os.path.join(base_dir, new_folder_name)
        
        # Rename the folder
        os.rename(old_folder_path, new_folder_path)
        
        # Increment the new index
        new_index += 1

print("Renaming completed.")
