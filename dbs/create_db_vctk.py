#tests and train are the same dbs here
import os
import random
import sqlite3
import sys
import yaml
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
current_dir = Path(__file__).resolve().parent
from db_modelvctk import gather_vctk_files,AudioDatabase


# Ensure the parent directory is in the sys.path for module imports

yaml_file_path = current_dir / '../constants.yaml'

# Load constants from YAML file
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)

# Extracting constants from the YAML config

root_data_folder = config['root_data_folder_vctk']
db_path_train = "audio_files_Train_vctk.db"
db_path_test = "audio_files_Test_vctk.db"

# Create database instances
audio_db_train = AudioDatabase(db_path_train)
audio_db_test = AudioDatabase(db_path_test)

# Gather and insert train files into the database
#1.in that case its all the files, need to seperate and then after seperation its ok
#check for the random
train_files = gather_vctk_files(root_data_folder, all_files=1,dataset_type = "train")
test_files = gather_vctk_files(root_data_folder,all_files=1,dataset_type="test")

audio_db_train.insert_files(train_files)
audio_db_test.insert_files(test_files)

# Gather and insert test files into the database
#test_files = gather_vctk_files(root_data_folder_test, all_files=1)
#audio_db_test.insert_files(test_files)

print("randfile 1")
print(audio_db_test.get_random_file(audio_db_test.get_random_speaker()))
print(len(audio_db_test.get_all_data()))
print("randfile 2")
print(audio_db_test.get_random_file(audio_db_test.get_random_speaker()))