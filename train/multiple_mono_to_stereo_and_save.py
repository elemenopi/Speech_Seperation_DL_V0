import librosa
import soundfile as sf
import numpy as np
import os
import json
# Input and output folders
input_folder = "../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/TESTS_1311/output00004"
output_folder = "input_for_separation_TEST"  # Update this to your desired output folder

# Number of channels (number of mono files to combine)
num_channels = 6

# Initialize a list to hold the audio data for each channel
y_n = []

json_file_name = os.path.join(input_folder,"metadata.json")
with open(json_file_name,'r') as f:
    json_file = json.load(f)


print(json_file)
sr = 44100

for i in range(num_channels):
    # Construct the file name with zero-padding
    file_name = os.path.join(input_folder, f"mic0{i}_mixed.wav")

    # Check if the file exists
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exist. Please check the file name and path.")
        continue  # Skip this file and proceed with the next
    
    # Load the audio file
    y,_ = librosa.load(file_name,sr = 44100, mono=True)
    print(len(y))
    print(sr)
    y_n.append(y)

# Ensure that at least one file was loaded
if len(y_n) == 0:
    print("No audio files were loaded. Exiting the script.")
    exit()

# Ensure all audio signals have the same length
min_len = min(len(y) for y in y_n)
y_n = [y[:min_len] for y in y_n]

# Stack the audio signals into an array of shape (num_channels, num_samples)
multi_channel_audio = np.stack(y_n, axis=0)  # Shape: (num_channels, num_samples)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Save the multi-channel audio file
# Note: soundfile.write expects data in shape (num_samples, num_channels), so we transpose
output_file = os.path.join(output_folder, 'multi_channel_output.wav')
sf.write(output_file, multi_channel_audio.T, sr)  # Transpose to (num_samples, num_channels)

print(f"Multi-channel audio file saved to {output_file}")
print(f"Audio shape: {multi_channel_audio.shape}")  # Should be (num_channels, num_samples)
