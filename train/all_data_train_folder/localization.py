
import argparse
import os

from collections import namedtuple

import librosa
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import yaml
import utils
from pathlib import Path

from visualization import draw_diagram
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
checkpoint_dir = os.path.join(checkpoint_dir,"cos")
checkpoint_dir = os.path.join(checkpoint_dir,"multimic_experiment")
pretrain_demucs = True
if pretrain_demucs == True:
    from model_for_demucs import CosNetwork
    from model_for_demucs import load_pretrain,center_trim,normalize_input,unnormalize_input
    print("pretrained")
else:
    from model import CosNetwork
    from model import load_pretrain,center_trim,normalize_input,unnormalize_input

from eval_utils import si_sdr


yaml_file_path = Path(__file__).resolve().parent.parent / 'constants.yaml'
with open(yaml_file_path, "r") as file:
        config_yaml = yaml.safe_load(file)
yaml_train_dir = config_yaml["train_folder_Dataloader"]
yaml_pretrain_path = config_yaml["pretrain_path"]
yaml_checkpoints_dir = config_yaml["checkpoints_path"]
yaml_learning_rate = config_yaml["learning_rate"]
yaml_start_epoch = config_yaml["start_epoch"]
yaml_batch_size = config_yaml["batch_size"]
yaml_epochs = config_yaml["epochs"]
name = "multimic_experiment"
ALL_WINDOW_SIZES = config_yaml["ALL_WINDOW_SIZES"]
FAR_FIELD_RADIUS = config_yaml["FAR_FIELD_RADIUS"]
# Constants which may be tweaked based on your setup
ENERGY_CUTOFF = 0.008
NMS_RADIUS = np.pi / 4
NMS_SIMILARITY_SDR = -7  # SDR cutoff for different candidates

CandidateVoice = namedtuple("CandidateVoice", ["angle", "energy", "data"])


def nms(candidate_voices, nms_cutoff):
    """
    Runs non-max suppression on the candidate voices
    """
    final_proposals = []
    initial_proposals = candidate_voices

    while len(initial_proposals) > 0:
        new_initial_proposals = []
        sorted_candidates = sorted(initial_proposals,
                                   key=lambda x: x[1],
                                   reverse=True)

        # Choose the loudest voice
        best_candidate_voice = sorted_candidates[0]
        final_proposals.append(best_candidate_voice)
        sorted_candidates.pop(0)

        # See if any of the rest should be removed
        for candidate_voice in sorted_candidates:
            different_locations = utils.angular_distance(
                candidate_voice.angle, best_candidate_voice.angle) > NMS_RADIUS

            # different_content = abs(
            #     candidate_voice.data -
            #     best_candidate_voice.data).mean() > nms_cutoff

            different_content = si_sdr(
                candidate_voice.data[0],
                best_candidate_voice.data[0]) < nms_cutoff

            if different_locations or different_content:
                new_initial_proposals.append(candidate_voice)

        initial_proposals = new_initial_proposals

    return final_proposals


def forward_pass(model, target_angle, mixed_data, conditioning_label, args):
    """
    Runs the network on the mixed_data
    with the candidate region given by voice
    """
    target_pos = np.array([
        FAR_FIELD_RADIUS * np.cos(target_angle),
        FAR_FIELD_RADIUS * np.sin(target_angle)
    ])

    data, _ = utils.shift_mixture(
        torch.tensor(mixed_data).to(args.device), target_pos, args.mic_radius,
        args.sr)
    data = data.float().unsqueeze(0)  # Batch size is 1

    # Normalize input
    data, means, stds = normalize_input(data)

    # Run through the model
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))

    output_signal = model(padded, conditioning_label)
    output_signal = center_trim(output_signal, data)

    output_signal = unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

    output_np = output_voices.detach().cpu().numpy()[0]
    
    # Combine channels to mono by averaging
    mono_output = np.mean(output_np, axis=0)

    # Compute RMS energy
    energy = librosa.feature.rms(y=mono_output).mean()

    return output_np, energy


def run_separation(mixed_data, model, args,
                   energy_cutoff=ENERGY_CUTOFF,
                   nms_cutoff=NMS_SIMILARITY_SDR): # yapf: disable
    """
    The main separation by localization algorithm
    """
    # Get the initial candidates
    num_windows = len(ALL_WINDOW_SIZES) if not args.moving else 3
    starting_angles = utils.get_starting_angles(ALL_WINDOW_SIZES[0])
    candidate_voices = [CandidateVoice(x, None, None) for x in starting_angles]

    # All steps of the binary search
    for window_idx in range(num_windows):
        if args.debug:
            print("---------")
        conditioning_label = torch.tensor(utils.to_categorical(
            window_idx, 5)).float().to(args.device).unsqueeze(0)

        curr_window_size = ALL_WINDOW_SIZES[window_idx]
        new_candidate_voices = []

        # Iterate over all the potential locations
        for voice in candidate_voices:
            output, energy = forward_pass(model, voice.angle, mixed_data,
                                          conditioning_label, args)

            if args.debug:
                print("Angle {:.2f} energy {}".format(voice.angle, energy))
                fname = "out{}_angle{:.2f}.wav".format(
                    window_idx, voice.angle * 180 / np.pi)
                # sf.write(os.path.join(args.writing_dir, fname), output[0],
                #          args.sr)

            # If there was something there
            if energy > energy_cutoff:

                #print("passed")
                # We're done searching so undo the shifts
                if window_idx == num_windows - 1:
                    target_pos = np.array([
                        FAR_FIELD_RADIUS * np.cos(voice.angle),
                        FAR_FIELD_RADIUS * np.sin(voice.angle)
                    ])
                    unshifted_output, _ = utils.shift_mixture(output,
                                                           target_pos,
                                                           args.mic_radius,
                                                           args.sr,
                                                           inverse=True)
                    new_candidate_voices.append(
                        CandidateVoice(voice.angle, energy, unshifted_output))

                # Split region and recurse.
                # You can either split strictly (fourths)
                # or with some redundancy (thirds)
                else:
                    #new_candidate_voices.append(
                    #    CandidateVoice(
                    #        voice.angle + curr_window_size / 3,
                    #        energy, output))
                    #new_candidate_voices.append(
                    #    CandidateVoice(
                    #        voice.angle - curr_window_size / 3,
                    #        energy, output))
                    #new_candidate_voices.append(
                    #    CandidateVoice(
                    #        voice.angle,
                    #        energy, output))
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle + curr_window_size / 4,
                            energy, output))
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle - curr_window_size / 4,
                            energy, output))

        candidate_voices = new_candidate_voices

    # Run NMS on the final output and return
    return nms(candidate_voices, nms_cutoff)


def main():
    
    class Args:
        model_checkpoint = os.path.join(checkpoint_dir, "42_Cos_bg_0610.pt")
        input_file = os.path.join("input_for_separation", "exp_2111_3.wav")
        #print(input_file)
        output_dir = 'results_exp_2111_3_localization'
        sr = 44100  # Sampling rate
        n_channels = 6  # Number of channels
        use_cuda = True  # Set to True if using CUDA
        debug = True  # Set to True for saving intermediate outputs
        mic_radius = 0.0463  # Radius of the mic array
        duration = 3.0  # Duration in seconds for input to the network
        moving = False  # Set to True if the sources are moving

    args = Args()

    device = torch.device('cuda:1') if args.use_cuda else torch.device('cpu')

    args.device = device
    model = CosNetwork(n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device), strict=True)
    model.train = False
    model.to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mixed_data = librosa.core.load(args.input_file, mono=False, sr=args.sr)[0]# for regular
    #mixed_data = librosa.core.load(args.input_file, mono=False, sr=16000)[0]#if microphones used
    #mixed_data = librosa.resample(mixed_data, orig_sr=16000, target_sr=args.sr)#if microphones used
    #mixed_data = mixed_data / np.max(np.abs(mixed_data))#if microphones used??
    output_resampled_path = os.path.join(os.getcwd(), "resampled_output.wav")
    assert mixed_data.shape[0] == args.n_channels
    sf.write(output_resampled_path, mixed_data[0], args.sr)
    #print(mixed_data)
    assert mixed_data.shape[0] == args.n_channels

    temporal_chunk_size = int(args.sr * args.duration)#if duration>mixed_data length then seperate to chunks
    num_chunks = (mixed_data.shape[1] // temporal_chunk_size) + 1
    print("number of chunks:")
    print(num_chunks)
    for chunk_idx in range(num_chunks):
        curr_writing_dir = os.path.join(args.output_dir,
                                        "{:03d}".format(chunk_idx))
        if not os.path.exists(curr_writing_dir):
            os.makedirs(curr_writing_dir)

        args.writing_dir = curr_writing_dir
        curr_mixed_data = mixed_data[:, (chunk_idx *
                                         temporal_chunk_size):(chunk_idx + 1) *
                                     temporal_chunk_size]

        output_voices = run_separation(curr_mixed_data, model, args)
        for voice in output_voices:
            fname = "output_angle{:.2f}.wav".format(
                voice.angle * 180 / np.pi)
            sf.write(os.path.join(args.writing_dir, fname), voice.data[0],
                     args.sr)

        candidate_angles = [voice.angle for voice in output_voices]
        diagram_window_angle = ALL_WINDOW_SIZES[2] if args.moving else ALL_WINDOW_SIZES[-1]
        draw_diagram([], candidate_angles,
                     diagram_window_angle,
                     os.path.join(args.writing_dir, "positions.png".format(chunk_idx)))

if __name__ == '__main__':
    main()