import numpy as np
import utils
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.io.wavfile import read,write
import json
import random
import torch
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, oaconvolve,stft
import time
from custom_dataset_bg import customdataset
import torch.optim as optim
from torch.utils.data import DataLoader
import soundfile as sf
import torch.nn.functional as F
pretrain_demucs = True
if pretrain_demucs == True:
    from model_for_demucs import CosNetwork
    from model_for_demucs import load_pretrain,center_trim,normalize_input,unnormalize_input
    print("pretrained")
else:
    from model import CosNetwork
    from model import load_pretrain,center_trim,normalize_input,unnormalize_input
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
use_cuda = torch.cuda.is_available()
ds = customdataset("../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Cuppa_tests/", negatives=1.0)
dl = DataLoader(ds,batch_size=1,shuffle=True)
model = CosNetwork()
device = torch.device('cuda:1' if use_cuda else 'cpu')
model.to(device)

checkpoints_dir = "checkpoints/cos"
name = "multimic_experiment"
print("station 0 ")
checkpoints_dir = os.path.join(checkpoints_dir,name)
checkpoint_path = os.path.join(checkpoints_dir, "42_Cos_bg_0610.pt")
print(checkpoint_path)
#checkpoint_path = "synthetic_6mics_.0725m_44100kHz.pt"
state_dict = torch.load(checkpoint_path, map_location=device)
print("station 1 ")
#url = 'https://dl.fbaipublicfiles.com/demucs/v3.0/demucs-e07c671f.th'#pretrain demucs
#state_dict = torch.hub.load_state_dict_from_url(url)#pretrain demucs
#print(state_dict.keys())

#exit()
load_pretrain(model, state_dict)
#model.load_state_dict(state_dict)
model.eval()
print("station 2 ")
print(count_parameters(model))
for (mixed_data,target_voice_data,window_idx_one_hot) in dl:
    mixed_data = mixed_data.to(device)
    target_voice_data = target_voice_data.to(device)
    window_idx_one_hot = window_idx_one_hot.to(device)
    print(len(mixed_data[0][0]))
    data, means, stds = normalize_input(mixed_data)
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))
    output_signal = model(padded, window_idx_one_hot)
    output_signal = center_trim(output_signal, data)
    output_signal = unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:, 0]
    loss = model.loss(output_voices,target_voice_data)
    print(loss.item())
    # Convert the first channel to a NumPy array and save as a WAV file
    output_voice_np = output_voices[0][0].detach().cpu().numpy()
    output_file_path = os.path.join(os.getcwd(), 'output_voice_4layer_sum.wav')
    target_voice_np = target_voice_data[0][0].detach().cpu().numpy()
    target_file_path = os.path.join(os.getcwd(), 'target_voice_4layer_sum.wav')
    mixed_file_path = os.path.join(os.getcwd(), 'mixed_voice_4layer_sum.wav')
    mixed_voice_np = mixed_data[0][0].detach().cpu().numpy()
    print(mixed_data.shape)
       # Normalize signals
    output_voice_norm = output_voice_np / np.max(np.abs(output_voice_np))
    target_voice_norm = target_voice_np / np.max(np.abs(target_voice_np))
    mixed_voice_norm = mixed_voice_np / np.max(np.abs(mixed_voice_np))
    sf.write(output_file_path, output_voice_np, 44100)#not normalized
    sf.write(target_file_path,target_voice_np,44100)
    sf.write(mixed_file_path,mixed_voice_np,44100)
     # Plot output vs. target
    #plt.figure(figsize=(12, 6))
    #plt.plot(mixed_voice_norm, label='Mixed Voice (Normalized)')
    #plt.plot(output_voice_norm, label='Output Voice (Normalized)')
    #
    #plt.plot(target_voice_norm, label='Target Voice (Normalized)')
    #plt.title('Normalized Output Voice vs Target Voice vs Mixed')
    #plt.xlabel('Sample Index')
    #plt.ylabel('Normalized Amplitude')
    #plt.legend()
    #
    #plot_path = os.path.join(os.getcwd(), 'output_vs_target_plot2.png')
    #plt.savefig(plot_path)
    #plt.close()
    #print(f"Saved plot to {plot_path}")
#
#
    ## Compute and plot STFTs
    #_, _, output_spectrogram = stft(output_voice_np, fs=44100, nperseg=1024)
    #_, _, target_spectrogram = stft(target_voice_np, fs=44100, nperseg=1024)
    ## Convert to decibels
    #output_spectrogram_dB = 20 * np.log10(np.abs(output_spectrogram) + 1e-8)
    #target_spectrogram_dB = 20 * np.log10(np.abs(target_spectrogram) + 1e-8)
    #
    ## Plot Output Voice Spectrogram
    #plt.figure(figsize=(12, 6))
    #plt.imshow(output_spectrogram_dB, aspect='auto', origin='lower', cmap='viridis')
    #plt.title("Output Voice Spectrogram (dB)")
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    #plt.colorbar(label="Amplitude (dB)")
    #output_spec_path = os.path.join(os.getcwd(), 'output_voice_spectrogram_dB.png')
    #plt.savefig(output_spec_path)
    #plt.close()
    #print(f"Saved output spectrogram to {output_spec_path}")
    #
    ## Plot Target Voice Spectrogram
    #plt.figure(figsize=(12, 6))
    #plt.imshow(target_spectrogram_dB, aspect='auto', origin='lower', cmap='viridis')
    #plt.title("Target Voice Spectrogram (dB)")
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    #plt.colorbar(label="Amplitude (dB)")
    #target_spec_path = os.path.join(os.getcwd(), 'target_voice_spectrogram_dB.png')
    #plt.savefig(target_spec_path)
    #plt.close()
    #print(f"Saved target spectrogram to {target_spec_path}")

    # Plot mixed, output, and target voice waveforms in the time domain in separate subplots
    plt.figure(figsize=(12, 10))

    # Plot Mixed Voice
    plt.subplot(3, 1, 1)
    plt.plot(mixed_voice_norm)
    plt.title('Mixed Voice (Normalized)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # Plot Output Voice
    plt.subplot(3, 1, 2)
    plt.plot(output_voice_norm)
    plt.title('Output Voice (Normalized)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # Plot Target Voice
    plt.subplot(3, 1, 3)
    plt.plot(target_voice_norm)
    plt.title('Target Voice (Normalized)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # Save the figure
    plot_path = os.path.join(os.getcwd(), 'output_vs_target_separate_plot2.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved time-domain plots in separate subplots to {plot_path}")


    # Compute STFTs for output and target voices
    _, _, output_spectrogram = stft(output_voice_np, fs=44100, nperseg=1024)
    _, _, target_spectrogram = stft(target_voice_np, fs=44100, nperseg=1024)

    # Convert to decibels
    output_spectrogram_dB = 20 * np.log10(np.abs(output_spectrogram) + 1e-8)
    target_spectrogram_dB = 20 * np.log10(np.abs(target_spectrogram) + 1e-8)

    # Plot both spectrograms in one figure with two subplots
    plt.figure(figsize=(12, 10))

    # Output Spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(output_spectrogram_dB, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Output Voice Spectrogram (dB)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Amplitude (dB)")

    # Target Spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(target_spectrogram_dB, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Target Voice Spectrogram (dB)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Amplitude (dB)")

    # Save the spectrogram figure
    spectrogram_plot_path = os.path.join(os.getcwd(), 'combined_spectrograms_dB.png')
    plt.tight_layout()
    plt.savefig(spectrogram_plot_path)
    plt.close()
    print(f"Saved combined spectrograms to {spectrogram_plot_path}")
    break