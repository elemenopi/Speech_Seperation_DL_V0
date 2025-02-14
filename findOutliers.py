import os
import json
import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Base directory containing the folders
base_dir = "gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds_2d/"

# Function to extract features from a given audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
    return np.mean(mfccs, axis=1)  # Return the mean MFCCs as feature vector

# Function to analyze each folder and detect outliers
def find_outliers_in_folders(base_dir):
    folder_paths = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if folder.startswith('output')]
    all_features = []
    metadata_positions = []

    for folder in folder_paths:
        # Load metadata for this folder
        metadata_path = os.path.join(folder, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as meta_file:
                metadata = json.load(meta_file)

        for wav_file in os.listdir(folder):
            if wav_file.endswith(".wav"):
                file_path = os.path.join(folder, wav_file)
                features = extract_features(file_path)
                all_features.append(features)

                # If available, add position data from metadata
                voice_key = wav_file.split("_")[1].replace(".wav", "")
                if voice_key in metadata:
                    position = metadata[voice_key]["Position"]
                    metadata_positions.append(position)

    # Convert features to a numpy array
    all_features = np.array(all_features)

    # Detect outliers using Z-score method
    z_scores = zscore(all_features, axis=0)
    outlier_mask_zscore = np.abs(z_scores) > 3  # Adjust threshold if needed

    # Detect outliers using Isolation Forest
    iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination if needed
    outlier_mask_iforest = iso_forest.fit_predict(all_features) == -1

    # Combine results
    combined_outliers = np.logical_or(outlier_mask_zscore.any(axis=1), outlier_mask_iforest)

    # Visualize or output the results
    outlier_indices = np.where(combined_outliers)[0]
    print(f"Found {len(outlier_indices)} potential outliers.")

    # Optional: Visualization
    plt.scatter(np.array(metadata_positions)[:, 0], np.array(metadata_positions)[:, 1], c=combined_outliers, cmap='coolwarm')
    plt.title("Speaker Positions with Outliers Highlighted")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.show()

    # Return the outlier folder names for further inspection
    return [folder_paths[i] for i in outlier_indices]

# Run the analysis
outlier_folders = find_outliers_in_folders(base_dir)
print("Outlier Folders:")
for folder in outlier_folders:
    print(folder)
