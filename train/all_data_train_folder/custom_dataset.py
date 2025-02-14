import sys
import os
from typing import Tuple
from scipy.io.wavfile import read,write
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath('..'))
import torch
import torchvision
import numpy as np
from pathlib import Path
import soundfile as sf
import json
import random
import utils
import librosa
#from RIRnewv.constants import FAR_FIELD_RADIUS,ALL_WINDOW_SIZES
from data_augmentation import RandomAudioPerturbation
import matplotlib.pyplot as plt
import yaml
yaml_file_path = Path(__file__).resolve().parent.parent / 'constants.yaml'
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)
ALL_WINDOW_SIZES = config["ALL_WINDOW_SIZES"]
FAR_FIELD_RADIUS = config["FAR_FIELD_RADIUS"]


class customdataset(torch.utils.data.Dataset):
    #def __init__(self,input_dir,n_mics = 6,sr = 44100,perturb_prob = 0.0,window_idx = 2,negatives = 0.0,mic_radius = 0.03231 ):
    #    super().__init__()
    #    print(input_dir)
    #    self.dirs = sorted(list(Path(input_dir).glob('*[0-9]')))
    #    #print(self.dirs)
    #    self.n_mics = n_mics
    #    self.sr = sr
    #    self.mic_radius = mic_radius
#
    #    self.perturb_prob = perturb_prob
    #    
    #    self.negatives = negatives
    #    self.window_idx = window_idx
    #def __len__(self):
    #    return len(self.dirs)
    #def __getitem__(self,idx:int):
    #    num_windows = len(ALL_WINDOW_SIZES)
    #    if self.window_idx == -1:
    #        curr_window_idx = np.random.randint(0,5)
    #    else:
    #        curr_window_idx = self.window_idx
    #    curr_window_size = ALL_WINDOW_SIZES[curr_window_idx]
    #    #get all angles of the window size around unit circle
    #    candidate_angles = utils.get_starting_angles(curr_window_size)
    #    #get directory name from index
    #    curr_dir = self.dirs[idx]
    #    #print("the current dir")
    #    #print(curr_dir)
    #    #take the information about the data in the directory from metadata
    #    with open(Path(curr_dir)/'metadata.json') as json_file:
    #        metadata = json.load(json_file)
    #    
    #    if np.random.uniform()<self.negatives:
    #        #returns example out of the region
#
    #        target_angle = self.get_negative_region(metadata,candidate_angles)
    #    else:
    #        #returns an example in the region
    #        #positive example target angle
    #        #takes a random voice and the candidate angles (possible windows)
    #        #returns the window angle closest to the candidate angle
    #        target_angle = self.get_positive_region(metadata,candidate_angles)
    #        print("the target angle")
    #        print(target_angle)
    #    all_sources, target_voice_data = self.get_mixture_and_gt(metadata,curr_dir,target_angle,curr_window_size)
    #    all_sources = torch.stack(all_sources,dim = 0)# [src1,src2,...],srci = [fval,....]
    #    mixed_data = torch.sum(all_sources,dim = 0)
#
    #    
    #    target_voice_data = torch.stack(target_voice_data,dim = 0)
    #    target_voice_data = torch.sum(target_voice_data,dim = 0)
    #    window_idx_one_hot = torch.tensor(utils.to_categorical(curr_window_idx,num_windows)).float()
#
    #    return (mixed_data,target_voice_data,window_idx_one_hot) 
    #
    #
    #def get_mixture_and_gt(self,metadata,curr_dir,target_angle,curr_window_size):
    #    target_pos = np.array([FAR_FIELD_RADIUS * np.cos(target_angle),FAR_FIELD_RADIUS * np.sin(target_angle)])
    #    #random_perturb = RandomAudioPerturbation()
#
    #    all_sources = []
    #    target_voice_data = []
    #    
    #    gt_audio_files = sorted(list(Path(curr_dir).rglob("*bg*" + ".wav")))
    #    if len(gt_audio_files):
    #        gt_waveforms = []
    #        for _,gt_audio_file in enumerate(gt_audio_files):
    #            gt_waveform,_ = librosa.core.load(gt_audio_file,sr = self.sr,mono = True)
    #            gt_waveforms.append(torch.from_numpy(gt_waveform))
    #            shifted_gt,_ = utils.shift_mixture(np.stack(gt_waveforms),target_pos,self.mic_radius,self.sr)
#
    #        perturbed_source = torch.tensor(shifted_gt).float()
    #        all_sources.append(perturbed_source)
#
#
    #    for key in metadata.keys():
    #        #if "bg" in key and not bg_done:
    #        #    key = "bg"
    #        #    bg_done = 1
    #        #elif "bg" in key and bg_done:
    #        #    continue
    #        if "bg" in key:
    #            continue
    #        gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + "*" + ".wav")))
#
    #        #gt_audio_files = [file for file in gt_audio_files if ('voice' in file.stem and key in file.stem)]
 #
    #        assert len(gt_audio_files) > 0 , "no files found"
    #        gt_waveforms = []
    #        for _,gt_audio_file in enumerate(gt_audio_files):
    #            gt_waveform,_ = librosa.core.load(gt_audio_file,sr = self.sr,mono = True)
    #            gt_waveforms.append(torch.from_numpy(gt_waveform))
    #            shifted_gt,_ = utils.shift_mixture(np.stack(gt_waveforms),target_pos,self.mic_radius,self.sr)
#
#
#
    #        perturbed_source = torch.tensor(shifted_gt).float()
    #        all_sources.append(perturbed_source)
    #        if "bg" in key:
    #            continue
    #        locs_voice = metadata[key]["Position"]
    #        #voice_angle = np.arctan2(locs_voice[1],locs_voice[0])
    #        voice_angle = locs_voice[1]
    #        #todo : take into account front back confusion
    #        #print("the voice angle from metadata")
    #        #print(voice_angle)
    #        #print("for the key")
    #        #print(key)
#
    #        if abs(voice_angle - target_angle)<(curr_window_size/2):
    #            target_voice_data.append(perturbed_source.view(perturbed_source.shape[0],perturbed_source.shape[1]))
#
    #        else:
    #            target_voice_data.append(torch.zeros((perturbed_source.shape[0],perturbed_source.shape[1])))
    #    return all_sources,target_voice_data
#
#
    #def get_positive_region(self,metadata,candidate_angles):
    #    voice_keys = [x for x in metadata if "voice" in x]
    #    random_key = random.choice(voice_keys)
    #    voice_pos = metadata[random_key]["Position"]
 #
    #    voice_angle = voice_pos[1]
#
    #    angle_idx = (np.abs(candidate_angles - voice_angle)).argmin()
    #    target_angle = candidate_angles[angle_idx]
    #    #("chosen positive, target angle")
    #    #print(target_angle)
    #    print(f"voice angle : {voice_angle}, target angle : {target_angle} , voice key: {random_key}")
    #    return target_angle
    #
    #def get_negative_region(self,metadata,candidate_angles):
    #    voice_keys = [x for x in metadata if "voice" in x]
    #    random_key = random.choice(voice_keys)
    #    voice_pos = metadata[random_key]["Position"]
    #    voice_angle = voice_pos[1]
    #    angle_idx = (np.abs(candidate_angles-voice_angle)).argmin()
#
    #    p = np.zeros_like(candidate_angles)
    #    for i in range(p.shape[0]):
    #        if i == angle_idx:
    #            p[i] = 0
    #        else:
    #            dist = min(abs(i-angle_idx),(len(candidate_angles) - angle_idx + i))
    #            p[i] = 1/(dist)
    #    p/=p.sum()
    #    matching_shift = True
    #    voice_pos = np.array([voice_pos[0]*np.cos(voice_pos[1]),voice_pos[0]*np.sin(voice_pos[1])])
    #    _,true_shift = utils.shift_mixture(np.zeros((self.n_mics,10)),voice_pos,self.mic_radius,self.sr)
    #    while matching_shift:
    #        #choose a close but not target angle
    #        target_angle = np.random.choice(candidate_angles,p = p)
    #        #choose random position for shifting
    #        random_pos = np.array([FAR_FIELD_RADIUS*np.cos(target_angle),
    #                               FAR_FIELD_RADIUS*np.sin(target_angle)])
    #        _,curr_shift = utils.shift_mixture(np.zeros((self.n_mics,10)),random_pos,self.mic_radius,self.sr)
    #        if true_shift!=curr_shift:
    #            matching_shift = False
    #    #print("chosen negative, target angle")
    #    #print(target_angle)
    #    return target_angle
    def __init__(self, input_dir, n_mics=6, sr=44100, perturb_prob=0.1,
                 window_idx=-1, negatives=0.2, mic_radius=0.0463):
        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('*[0-9]')))

        # Physical params
        self.n_mics = n_mics
        self.sr = sr
        self.mic_radius = mic_radius

        # Data augmentation
        self.perturb_prob = perturb_prob

        # Training params
        self.negatives = negatives  # Fraction of negatives in training
        self.window_idx = window_idx  # Set to -1 to pick randomly

    def __len__(self) -> int:
        return len(self.dirs)
    def check_properties_and_print(self,metadata,curr_window_size,angle,curr_dir):
        print(f"curr_dir {curr_dir}")
        print(f"metadata {metadata}")
        print(f"curr_window_size {curr_window_size}")
        print(f"angle { angle}")
        utils.plot_area(metadata,curr_window_size,angle)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        num_windows = len(ALL_WINDOW_SIZES)
        if self.window_idx == -1:
            curr_window_idx = np.random.randint(0, 5)
        else:
            curr_window_idx = self.window_idx

        curr_window_size = ALL_WINDOW_SIZES[curr_window_idx]
        candidate_angles = utils.get_starting_angles(curr_window_size)

        curr_dir = self.dirs[idx]

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            metadata = json.load(json_file)

        # Random split of negatives and positives
        if np.random.uniform() < self.negatives:
            target_angle = self.get_negative_region(metadata, candidate_angles)
        else:
            target_angle = get_positive_region(metadata, candidate_angles)

        all_sources, target_voice_data = self.get_mixture_and_gt(
            metadata, curr_dir, target_angle, curr_window_size)

        # Mixture
        all_sources = torch.stack(all_sources, dim=0)
        mixed_data = torch.sum(all_sources, dim=0)

        # GTs
        target_voice_data = torch.stack(target_voice_data, dim=0)
        target_voice_data = torch.sum(target_voice_data, dim=0)

        window_idx_one_hot = torch.tensor(
            utils.to_categorical(curr_window_idx, num_windows)).float()
        
        
        #debugging

        # Save the mixed data and target voice data as wav files
        
            # Convert to NumPy arrays
        #mixed_data_np = mixed_data.cpu().numpy()
        #target_voice_data_np = target_voice_data.cpu().numpy()
#
        ## Assuming the data is in the range [-1, 1], scale to int16
        #mixed_data_np = np.clip(mixed_data_np * 32767, -32768, 32767).astype(np.int16)
        #target_voice_data_np = np.clip(target_voice_data_np * 32767, -32768, 32767).astype(np.int16)
#
        ## Get the directory where the script is located
        #script_dir = Path(__file__).parent
#
        ## Paths to save the WAV files
        #mixed_wav_path = script_dir / f"mixed_data_{idx}.wav"
        #target_wav_path = script_dir / f"target_voice_data_{idx}.wav"
#
        ## Write the audio files
        #sf.write(str(mixed_wav_path), mixed_data_np.T, self.sr)  # Use .T to ensure correct shape (T, M)
        #sf.write(str(target_wav_path), target_voice_data_np.T, self.sr)
#
        ## Return the tensors as usual

        return (mixed_data, target_voice_data, window_idx_one_hot)

    def get_negative_region(self, metadata, candidate_angles):
        """Chooses a target angle which is adjacent to a voice region"""
        # Choose a random voice
        voice_keys = [x for x in metadata if "voice" in x]
        random_key = random.choice(voice_keys)
        voice_pos = np.array(metadata[random_key]["Position"])
        voice_angle = np.arctan2(voice_pos[1], voice_pos[0])
        angle_idx = (np.abs(candidate_angles - voice_angle)).argmin()

        # Non uniform distribution to prefer regions close to a voice
        p = np.zeros_like(candidate_angles)
        for i in range(p.shape[0]):
            if i == angle_idx:
                # Can't choose the positive region
                p[i] = 0
            else:
                # Regions close to the voice are weighted more
                dist = min(abs(i - angle_idx),
                           (len(candidate_angles) - angle_idx + i))
                p[i] = 1 / (dist)

        p /= p.sum()

        # Make sure we choose a region with different per-channel shifts from the voice
        matching_shift = True
        
        _, true_shift = utils.shift_mixture(np.zeros(
            (self.n_mics, 10)), voice_pos[:2], self.mic_radius, self.sr)
        while matching_shift:
            target_angle = np.random.choice(candidate_angles, p=p)
            random_pos = np.array([
                FAR_FIELD_RADIUS * np.cos(target_angle),
                FAR_FIELD_RADIUS * np.sin(target_angle)
            ])
            _, curr_shift = utils.shift_mixture(np.zeros(
                (self.n_mics, 10)), random_pos, self.mic_radius, self.sr)
            if true_shift != curr_shift:
                matching_shift = False

        return target_angle

    def get_mixture_and_gt(self, metadata, curr_dir, target_angle,
                           curr_window_size):
        """
        Given a target angle and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """
        target_pos = np.array([
            FAR_FIELD_RADIUS * np.cos(target_angle),
            FAR_FIELD_RADIUS * np.sin(target_angle)
        ])
        random_perturb = RandomAudioPerturbation()

        # Iterate over different sources
        all_sources = []
        target_voice_data = []
        bg_done = False
        for key in metadata.keys():
            if "bg" in key and bg_done == False:
                gt_audio_file = sorted(
                    list(Path(curr_dir).rglob("*bg*.wav"))
                )
                bg_done = True

            if "bg" in key and bg_done == True:
                continue
                        
            gt_audio_files = sorted(
                list(Path(curr_dir).rglob("*" + key + ".wav")))
            
            assert len(gt_audio_files) > 0, "No files found in {}".format(
                curr_dir)
            gt_waveforms = []

            # Iterate over different mics
            for _, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file, sr = self.sr,mono=True)
                gt_waveforms.append(torch.from_numpy(gt_waveform))
                
                shifted_gt, _ = utils.shift_mixture(np.stack(gt_waveforms),
                                                    target_pos[:2],
                                                    self.mic_radius, self.sr)
            

            #print(key)
            # Data augmentation
            if np.random.uniform() < self.perturb_prob:
                perturbed_source = torch.tensor(
                    random_perturb(shifted_gt)).float()
            else:
                perturbed_source = torch.tensor(shifted_gt).float()

            all_sources.append(perturbed_source)

            # Check which foregrounds are in the angle of interest
            if "bg" in key:
                continue

            locs_voice = metadata[key]['Position']
            voice_angle = np.arctan2(locs_voice[1], locs_voice[0])

            # Voice is inside our target area. Need to save for ground truth
            if abs(voice_angle - target_angle) < (curr_window_size / 2):
                target_voice_data.append(
                    perturbed_source.view(perturbed_source.shape[0],
                                          perturbed_source.shape[1]))

            # Train with front back confusion for 2 mics
            elif self.n_mics == 2 and abs(-voice_angle - target_angle) < (
                    curr_window_size / 2):
                target_voice_data.append(
                    perturbed_source.view(perturbed_source.shape[0],
                                          perturbed_source.shape[1]))

            # Voice is not within our region. Add silence
            else:
                target_voice_data.append(
                    torch.zeros((perturbed_source.shape[0],
                                 perturbed_source.shape[1])))

        return all_sources, target_voice_data


def get_positive_region(metadata, candidate_angles):
    """Chooses a target angle containing a voice region"""
    # Choose a random voice
    voice_keys = [x for x in metadata if "voice" in x]
    random_key = random.choice(voice_keys)
    voice_pos = metadata[random_key]["Position"]
    voice_pos = np.array(voice_pos)
    voice_angle = np.arctan2(voice_pos[1], voice_pos[0])
    
    # Get the sector closest to that voice
    angle_idx = (np.abs(candidate_angles - voice_angle)).argmin()
    target_angle = candidate_angles[angle_idx]

    return target_angle