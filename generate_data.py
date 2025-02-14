import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io.wavfile import read,write
from scipy.signal import convolve,fftconvolve
import time
import soundfile as sf
import random
import math
import librosa
import numpy as np
import json
import pyroomacoustics as pra
from scipy.spatial.distance import euclidean
import utils
import tqdm
import argparse
from pathlib import Path
import multiprocessing.dummy as mp

from concurrent.futures import ProcessPoolExecutor
import yaml
from dbs.db_model import AudioDatabase, gather_wav_files
yaml_file_path = Path(__file__).resolve().parent / 'constants.yaml'
# Load constants from YAML file
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)

# Extracting constants from the YAML config
SPEED_OF_SOUND = config['SPEED_OF_SOUND']
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4

BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5
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
parser = argparse.ArgumentParser(description='Generate room audio and output files.')
parser.add_argument('--output_dir_train', required=True, help='Path to the output directory for training data')
args = parser.parse_args()

# Use the passed argument instead of the hardcoded path
output_dir_train = args.output_dir_train#doesnt matter if train or test
train_test = "test"# does matter
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
def generate_mic_array_3d(room, roomDims,mic_radius: float, n_mics: int, height):
    microphones = []
    mic_x = roomDims[0]/2
    mic_y = roomDims[1]/2
    mic_z = height
    angular_difference_degrees = 60
    for i in range(n_mics):
        angle = math.radians(i * angular_difference_degrees)
        x = mic_x + mic_radius * math.cos(angle)
        y = mic_y + mic_radius * math.sin(angle)
        microphones.append([x,y,mic_z])
    microphones = np.array(microphones).transpose()
    room.add_microphone_array(microphones)
class data_generator:
    def __init__(self):
        self.speaker_soundfile_map = None
        self.audio_db_train = None
        self.audio_db_train = None
    def generateRoom(self):
        self.setDatabase()
        
        self.num_of_speakers = 2#random.randint(1,4)
        
        self.get_random_sounds(self.num_of_speakers,train_test)
        self.generate_channels_V2(bg =True,snr=random.randint(8,10),sir = np.random.uniform(0, 5.0))
    def setDatabase(self):
        db_path_train = "dbs//audio_files_Train_vctk.db"
        db_path_test = "dbs//audio_files_Test_vctk.db"
        self.audio_db_train = AudioDatabase(db_path_train)
        self.audio_db_test = AudioDatabase(db_path_test)

    def Generate_bg_signals(self, mic_array_height, fs, total_seconds=3):
        # Set up the room with the specified absorption and maximum order
        roomDims = [round(random.uniform(20, 40), 4)
                        ,round(random.uniform(20, 40), 4)
                        ,round(random.uniform(3, 4), 4)]
        
        absorption = np.random.uniform(low=0.6, high=0.99)
        room = pra.ShoeBox(
            roomDims,
            fs=fs,
            max_order=10,
            materials=pra.Material(absorption)
        )

        # Generate the microphone array in the room
        generate_mic_array_3d(
            room,
            roomDims=roomDims,
            mic_radius=0.0463,
            n_mics=6,
            height=mic_array_height
        )

        # Define the four corners of the room
        bg_loc1 = [0.2, 0.2, 1.4]
        bg_loc2 = [roomDims[0] - 0.2, roomDims[1] - 0.2, 1.4]
        bg_loc3 = [roomDims[0] - 0.2, 0.2, 1.4]
        bg_loc4 = [0.2, roomDims[1] - 0.2, 1.4]
        bg_locs = [bg_loc1, bg_loc2, bg_loc3, bg_loc4]

        # Select 4 random speakers from train or test set
        if train_test.lower() == "train":
            speakers = [self.audio_db_train.get_random_speaker() for _ in range(4)]
            soundfiles = [self.audio_db_train.get_random_file(speaker) for speaker in speakers]
        else:
            speakers = [self.audio_db_test.get_random_speaker() for _ in range(4)]
            soundfiles = [self.audio_db_test.get_random_file(speaker) for speaker in speakers]

        # Read and process the audio signals for each speaker
        total_samples = int(total_seconds * fs)
        signal_total = []
        
        for soundfile in soundfiles:
            signal, fs_read = librosa.load(soundfile, sr=44100)
            #print(fs_read)
            bg_length = len(signal)
            #print(bg_length)

            if bg_length < total_samples:
                # Repeat the signal to make it long enough
                signal = np.tile(signal, (total_samples // bg_length) + 1)

            # Now `bg_length` is guaranteed to be at least `total_samples`
            bg_start_idx = np.random.randint(len(signal) - total_samples)
            signal = signal[bg_start_idx:bg_start_idx + total_samples]
            signal_total.append(signal)
        #print("hello")
        # Add each speaker as a source in the room at the corner locations
        for idx in range(4):
            room.add_source(bg_locs[idx], signal=signal_total[idx])

    
        # Simulate the room acoustics and retrieve individual source contributions
        # 'premix' will have shape (num_mics, num_sources, num_samples)
        premix = room.simulate(return_premix=True)
        
        # Reorder premix to have shape (num_speakers, num_mics, num_samples)
        #bg_signals = np.transpose(premix, (1, 0, 2))

        return bg_locs, premix, speakers



    def Generate_bg_signals_old(self,room_dim,fs ,max_order ,rt60tgt ,total_seconds = 3):
        mixture = []
        e_absorption,max_order = pra.inverse_sabine(rt60tgt,room_dim)
        e_absorption = 0.4
        room = pra.ShoeBox(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
        #generate microphones for background room
        generate_mic_array_3d(room,roomDims=room_dim,mic_radius=0.0463,n_mics = 6,height =np.random.uniform(low=1.4, high=1.5) )
        #set locations in the corners
        bg_loc1 = [0.2,0.2,1.4]
        bg_loc2 = [room_dim[0]-0.2,room_dim[1]-0.2,1.4]
        bg_loc3 = [room_dim[0]-0.2,0.2,1.4]
        bg_loc4 = [0.2,room_dim[1]-0.2,1.4]
        bg_loc_total = [bg_loc1,bg_loc2,bg_loc3,bg_loc4]
        
        #set the sounds of bg speakers
        speakers = [self.audio_db_train.get_random_speaker() for i in range(4)]
        soundfiles = [self.audio_db_train.get_random_file(speaker) for speaker in speakers] 
        
        signal1,fs = sf.read(soundfiles[0])
        signal2,fs = sf.read(soundfiles[1])
        signal3,fs = sf.read(soundfiles[2])
        signal4,fs = sf.read(soundfiles[3])
        signal_total = [signal1,signal2,signal3,signal4]
        
        #extend to required total length in seconds
        for idx, s in enumerate(signal_total):
                total_samples = int(total_seconds * fs)
                s = s / np.abs(s).max()  # Normalization
                while len(s) < total_samples:
                    s = np.tile(s, 2)  # Double the length until it is long enough
                signal_total[idx] = s[:total_samples]  # Trim to the exact number of required samples

        for i in range(4):
            room.add_source(bg_loc_total[i],signal_total[i])
        #simulate and generate result
        room.simulate()
        
        rirs = room.rir
        ms50Idx = int(0.05*fs)
        total_signal = np.zeros((len(rirs), len(signal_total[0])))
        for mic_idx in range(len(rirs)):
            mic_signal = np.zeros(len(signal_total[0]))
            for speaker_idx in range(len(rirs[1])):
                speaker_rir = rirs[mic_idx][speaker_idx]#[:150000]
 
                #speaker_rir[:ms50Idx] = 0
                #speaker_rir = utils.highpass_filter(speaker_rir,50,fs)
                convolved_signal_rir = fftconvolve(signal_total[speaker_idx],speaker_rir,mode = "full")
                mic_signal+=convolved_signal_rir[:len(signal_total[0])]

            total_signal[mic_idx] = mic_signal
        
        return bg_loc_total,total_signal,speakers


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
        #print(self.speaker_soundfile_map)
        return self.speaker_soundfile_map
    def generate_channels_V2(self,rt60tgt = 0.7,snr = 8,sir = 1,duration = 3,bg = False):  
        try:
            roomDims = [round(random.uniform(min_room_x, max_room_x), 4)
                        ,round(random.uniform(min_room_y, max_room_y), 4)
                        ,round(random.uniform(min_room_z, max_room_z), 4)]
            voice_positions = []
            all_fg_signals = []
            #generating fg signals
            is_axis = True
            beta = 0
            for voice_idx, s in self.speaker_soundfile_map.items():
                audio, fs = librosa.load(s, sr=44100)
                #print(fs)
                max_abs_audio = np.abs(audio).max()
                if max_abs_audio > beta:
                    beta = max_abs_audio
            
            #print(beta)
            for voice_idx, s in self.speaker_soundfile_map.items():
                audio, fs = librosa.load(s, sr=44100)  # Again, librosa to ensure consistent sampling rate
                audio, _ = librosa.effects.trim(audio, top_db=20)
                if audio.std() == 0:
                    raise ValueError(f"Audio is silent after trimming for file: {s}")

                if is_axis:
                    is_axis = False
                    axisAudio = audio / beta
                    audio = axisAudio
                else:
                    audio = audio / beta
                    audio, _ = utils.get_mixed(axisAudio, audio, sir)

                total_samples = int(fs * duration)
                #print(f"the fs is {fs}")
                # Randomizing absorption within the desired range [0.4, 0.99]
                e_absorption = random.uniform(0.7, 0.99)
                max_order = 10  # Set max_order if needed or continue with pra.inverse_sabine
                
                room = pra.ShoeBox(roomDims, fs=fs, max_order=max_order, materials=pra.Material(e_absorption))
                mic_array_height = np.random.uniform(low=1.4, high=1.5)
                generate_mic_array_3d(room, roomDims,0.0463, 6 , mic_array_height)
                
                #voice_radius = np.random.uniform(low=1.0, high=5.0)
                #voice_theta = np.random.uniform(low=0, high=2 * np.pi)
                #voice_loc = [
                #    voice_radius * np.cos(voice_theta)+ roomDims[0]/2,
                #    voice_radius * np.sin(voice_theta)+ roomDims[1]/2,
                #    round(random.uniform(1.4, 1.5), 4)
                #]
                #voice_positions.append(voice_loc)


                check_distance = False
                while not check_distance:
                    #voice_radius = np.random.uniform(low=1.0, high=5.0)
                    voice_radius = np.random.uniform(low=1.0, high=3.0)
                    voice_theta = np.random.uniform(low=0, high=2 * np.pi)
                    voice_loc = [
                        voice_radius * np.cos(voice_theta)+ roomDims[0]/2,
                        voice_radius * np.sin(voice_theta) + roomDims[1]/2,
                        np.random.uniform(low=1.4, high=1.5)
                    ]
                    if all(euclidean(voice_loc,pos) >= 0.5 for pos in voice_positions):
                        voice_positions.append(voice_loc)
                        check_distance = True


                room.add_source(voice_loc, signal=audio)
                room.simulate()
                fg_signals = room.mic_array.signals[:,:total_samples]
                fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
                fg_signals = fg_signals * fg_target / abs(fg_signals).max()
                all_fg_signals.append(fg_signals)
                #print(all_fg_signals)
            #generating bg signals
            if False:
                
                bg_locs, bg_signals,bg_speakers = self.Generate_bg_signals(mic_array_height,fs ,total_seconds = 3)
                print(bg_locs)
                print(bg_speakers)
                bg_signals = np.array(bg_signals)
                bg_signals = bg_signals[:,:,:total_samples]
                bg_target = np.random.uniform(BG_VOL_MIN, BG_VOL_MAX)
                bg_signals = bg_signals * bg_target / abs(bg_signals).max()
                
                for idx,s in enumerate(all_fg_signals):
                    write(f"outputTest_fg{idx}.wav",44100,s[0])
                    
                write("outputTest_bg1.wav",  44100,bg_signals[0][0])
                write("outputTest_bg2.wav",  44100,bg_signals[1][0])
                write("outputTest_bg3.wav",  44100,bg_signals[2][0])
                write("outputTest_bg4.wav",  44100,bg_signals[3][0])
            
            is_axis = True
            alpha = 0
            # SNR calculations
            #the axis is used to normalize one axis sound for snr and normlize all the rest of backgrounds for microphones according to the same snr
            #if bg:
            #    if snr != 0:
            #        #bg_total = np.sum(bg_signals, axis=0)
            #        max_mixed_with_bg = None
            #        #print("all fg signals len")
            #        #print(len(all_fg_signals))
            #        #print("channels")
            #        #print(len(all_fg_signals[0]))
            #        for i in range(6):  # for each microphone
            #            bg_total = bg_signals[i]
            #            fg = []
            #            for voice_idx in range(self.num_of_speakers):
            #                fg.append(all_fg_signals[voice_idx][i])
            #            mixed_signal = np.sum(fg, axis=0)
            #            current_mixed_with_bg = mixed_signal + bg_total
#
            #            if max_mixed_with_bg is None:
            #                max_mixed_with_bg = current_mixed_with_bg
            #            else:
            #                max_mixed_with_bg = np.maximum(max_mixed_with_bg, current_mixed_with_bg)
            #            if is_axis == True:
            #                bg_signals[i],alpha = utils.get_mixed(mixed_signal, bg_total, snr)
            #                is_axis = False
            #            else:
            #                bg_signals[i] = bg_signals[i]*alpha

            #if bg:
            #    beta = 1 / np.max(np.abs(max_mixed_with_bg))

            
            
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
                #if bg == True:
                #    bg_buffer = np.pad(bg_signals[mic_idx],(0,total_samples))[:total_samples]
                #    bg_buffer = bg_buffer*beta # n_tilda_tilda
                #    write(output_prefix + f"bg{mic_idx}.wav",fs,bg_buffer)
                #    write(output_prefix+"mixed.wav",fs,all_fg_buffer+bg_buffer)
                #if bg == False:
                #    write(output_prefix + "mixed.wav", fs,all_fg_buffer)
                
            metadata = {}
            for voice_idx,speaker_id in enumerate(self.speaker_soundfile_map.keys()):
                #r,theta = utils.convertCartesianToPolar(self.speaker_placements[voice_idx][0],self.speaker_placements[voice_idx][1],self.micArrayCenter[0],self.micArrayCenter[1])
                metadata['voice{:02d}'.format(voice_idx)] = {
                    'Position': [voice_positions[voice_idx][0] - roomDims[0]/2,voice_positions[voice_idx][1] - roomDims[1]/2,voice_positions[voice_idx][2]],
                    'speaker_id': speaker_id
                }
            #if bg == True:
            #    for i in range(4):
            #        metadata[f'bg{i}'] = {'position':[bg_locs[i][0],bg_locs[i][1],bg_locs[i][2]]}
            metadata_file = str(Path(latestFolder)/"metadata.json")
            #print(metadata)-
            with open(metadata_file,"w") as f:
                json.dump(metadata,f,indent = 4)
        except IndexError as e:
            print(f"Index Error ignored {e}")
        except ValueError as e:
            print(f"ValueError ignored in generate_channels_V2: {e}")
generator = data_generator()
for i in range(NUM_OF_ROOMS):
    generator.generateRoom()
    print(f"generating room no : {i}")

