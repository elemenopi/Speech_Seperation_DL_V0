import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io.wavfile import read,write
from scipy.signal import convolve,fftconvolve
import time
import soundfile as sf
import random
import math
import numpy as np
import json
import pyroomacoustics as pra
import utils
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import yaml
from dbs.db_model import AudioDatabase, gather_wav_files
yaml_file_path = Path(__file__).resolve().parent / 'constants.yaml'
# Load constants from YAML file
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4
# Extracting constants from the YAML config
SPEED_OF_SOUND = config['SPEED_OF_SOUND']
FAR_FIELD_RADIUS = config['FAR_FIELD_RADIUS']
ALL_WINDOW_SIZES = [np.array(value) for value in config['ALL_WINDOW_SIZES']]
NUM_OF_ROOMS = config['NUM_OF_ROOMS']
root_data_folder = config['root_data_folder']
min_room_x = config['min_room_x']
min_room_y = config['min_room_y']
min_room_z = config['min_room_z']
max_room_x = config['max_room_x']
max_room_y = config['max_room_y']
max_room_z = config['max_room_z']
output_dir_test = config['test_folder']
output_dir_train = "gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSecondsT"#"OUTPUTS_DSI/OUTPUTS_TRAIN/OUTPUTS_3_NOBG/"

train_test = "train"
def create_next_folder(base_folder, folder_prefix):
        # Get a list of existing folders in the base directory
        existing_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]

        # Find the latest folder with the specified prefix
        latest_folder = max((folder for folder in existing_folders if folder.startswith(folder_prefix)), default=None)
        
        if latest_folder is None:
            # If no folder found, create the first one
            new_folder_name = f"{folder_prefix}00000"
        else:
            # Extract the numeric part and increment it
            folder_number = int(latest_folder[len(folder_prefix):])
            new_folder_number = folder_number + 1
            new_folder_name = f"{folder_prefix}{new_folder_number:05d}"

        # Create the new folder
        new_folder_path = os.path.join(base_folder, new_folder_name)
        os.makedirs(new_folder_path)

        return new_folder_path
def generate_mic_array(room, mic_radius: float, n_mics: int):
    """
    Generate a list of Microphone objects

    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[0., 0.], M=n_mics, phi0=0, radius=mic_radius)
    room.add_microphone_array(pra.MicrophoneArray(R, 44100))

class data_generator:
    def __init__(self):
        self.num_of_speakers = random.randint(3,5)
        self.speaker_soundfile_map = None
        self.audio_db_train = None
        self.audio_db_train = None
    def generateRoom(self):
        self.setDatabase()
        self.get_random_sounds(self.num_of_speakers,train_test)
        self.generate_channels_V2(bg_recording = False,snr=random.randint(8,10))
    def setDatabase(self):
        db_path_train = "dbs//audio_files_Train_vctk.db"
        db_path_test = "dbs//audio_files_Test_vctk.db"
        self.audio_db_train = AudioDatabase(db_path_train)
        self.audio_db_test = AudioDatabase(db_path_test)
    def get_random_sounds(self,num_speakers,trainOrTest):
        random_sounds = {}

        if trainOrTest.lower() == "train":
            for _ in range(num_speakers):
                speaker = self.audio_db_train.get_random_speaker()
                sound_for_speaker = self.audio_db_train.get_random_file(speaker)
                random_sounds[speaker] = sound_for_speaker
        else:
            for _ in range(num_speakers):
                speaker = self.audio_db_test.get_random_speaker()
                sound_for_speaker = self.audio_db_test.get_random_file(speaker)
                random_sounds[speaker] = sound_for_speaker

        
        self.speaker_soundfile_map =random_sounds
        
        return self.speaker_soundfile_map
    def generate_channels_V2(self,rt60tgt = 0.7,snr = 8,sir = 1,bg_recording = True,duration = 3):  
        try:
            left_wall = np.random.uniform(low=-20, high=-15)
            right_wall = np.random.uniform(low=15, high=20)
            top_wall = np.random.uniform(low=15, high=20)
            bottom_wall = np.random.uniform(low=-20, high=-15)
            absorption = np.random.uniform(low=0.1, high=0.99)
            corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                            [   right_wall, top_wall], [right_wall, bottom_wall]]).T
            
            voice_positions = []
            all_fg_signals = []
            
            for voice_idx, s in self.speaker_soundfile_map.items():
                room = pra.Room.from_corners(corners,
                                     fs=44100,
                                     max_order=10,
                                     absorption=absorption)
                mic_array = generate_mic_array(room, 0.0725, 6)
                voice_radius = np.random.uniform(low=1.0, high=5.0)
                voice_theta = np.random.uniform(low=0, high=2 * np.pi)
                voice_loc = [
                    voice_radius * np.cos(voice_theta),
                    voice_radius * np.sin(voice_theta)
                ]
                voice_positions.append(voice_loc)
                audio,fs = sf.read(s)
                room.add_source(voice_loc, signal=audio)
                room.image_source_model()
                room.simulate()
                total_samples = int(fs*duration)
                
         
                fg_signals = room.mic_array.signals[:,:total_samples]
                fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
                fg_signals = fg_signals * fg_target / abs(fg_signals).max()
                all_fg_signals.append(fg_signals)
            
            outputDirectory = output_dir_train
            latestFolder = create_next_folder(outputDirectory,'output')
            for mic_idx in range(6):
                output_prefix = str(Path(latestFolder) / "mic{:02d}_".format(mic_idx))
                all_fg_buffer = np.zeros((total_samples))
                for voice_idx in range(self.num_of_speakers):
                    curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],(0,total_samples))[:total_samples]
                    write(output_prefix + "voice{:02d}.wav".format(voice_idx),  fs,curr_fg_buffer)#.astype(np.int16) )#32
                    all_fg_buffer+=curr_fg_buffer
                write(output_prefix + "mixed.wav", fs,all_fg_buffer)
                
            metadata = {}
            for voice_idx,speaker_id in enumerate(self.speaker_soundfile_map.keys()):
                #r,theta = utils.convertCartesianToPolar(self.speaker_placements[voice_idx][0],self.speaker_placements[voice_idx][1],self.micArrayCenter[0],self.micArrayCenter[1])
                metadata['voice{:02d}'.format(voice_idx)] = {
                    'Position': [voice_positions[voice_idx][0],voice_positions[voice_idx][1]],
                    'speaker_id': speaker_id
                }
            metadata_file = str(Path(latestFolder)/"metadata.json")
            with open(metadata_file,"w") as f:
                json.dump(metadata,f,indent = 4)
        except IndexError as e:
            print(f"Index Error ignored {e}")
        except ValueError as e:
            print(f"ValueError ignored in generate_channels_V2: {e}")
generator = data_generator()
for i in range(4):
    generator.generateRoom()
    print(f"generating room no : {i}")