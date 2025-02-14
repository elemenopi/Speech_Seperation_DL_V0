import json
import os
import shutil
import numpy as np
def remove_folders(folders):
    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"removed {folder}")
        except Exception as e:
            print(f"Error removing {folder}: e")

def find_folders_with_close_angles(directory):
    folders_to_remove = []
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory,folder_name)
        if os.path.isdir(folder_path):
            metadata_path = os.path.join(folder_path,'metadata.json')
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    allVoices = [key for key in data.keys() if 'voice' in key.lower()]
                    allAngles = []
                    for voice in allVoices:
                        x = data[voice]["Position"][0]
                        y = data[voice]["Position"][1]
                        angle = np.arctan2(y,x)
                        allAngles.append(angle*180/np.pi)
                    print(allAngles)
                    for i in range(len(allAngles)):
                        for j in range(i+1,len(allAngles)):
                            if abs(allAngles[i] - allAngles[j])<25:
                                folders_to_remove.append(folder_path)
                except json.JSONDecodeError as e:
                    print(f"Error reading {metadata_path}: {e}")
                    continue
    return folders_to_remove

def main():
    directory = 'gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/eval_sisdr_2410_bg'
    folders = find_folders_with_close_angles(directory)
    for folder in folders:
        print(folder)
    print(f"total folders without metadata.json : {len(folders)}")
    confirm = input("Do you want to delete the folders?").strip().lower()
    if confirm.lower() == 'yes':
        remove_folders(folders)
    else:
        print("folders were not deleted")
if __name__ == "__main__":
    main()
    