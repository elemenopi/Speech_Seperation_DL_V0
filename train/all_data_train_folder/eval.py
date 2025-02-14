import argparse
import json
import os

from pathlib import Path

import torch
import numpy as np
import librosa
import soundfile as sf
import tqdm

from eval_utils import find_best_permutation_prec_recall, compute_sdr
from utils import angular_distance, check_valid_dir
pretrain_demucs = True
if pretrain_demucs == True:
    from model_for_demucs import CosNetwork
    from model_for_demucs import load_pretrain,center_trim,normalize_input,unnormalize_input
    print("pretrained")
else:
    from model import CosNetwork
    from model import load_pretrain,center_trim,normalize_input,unnormalize_input
from localization import run_separation, CandidateVoice

import multiprocessing.dummy as mp
from multiprocessing import Lock



class Args:
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_dir = os.path.join(checkpoint_dir,"cos")
    checkpoint_dir = os.path.join(checkpoint_dir,"multimic_experiment")
    def __init__(self):
        self.test_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/eval_sisdr_2410_bg'  # Replace with actual path
        self.model_checkpoint = os.path.join(self.checkpoint_dir, "42_Cos_bg_0610.pt")  # Replace with actual model path
        self.sr = 44100
        self.n_channels = 6
        self.use_cuda = True
        self.debug = False
        self.mic_radius = 0.0463
        self.n_workers = 4
        self.n_voices = 2
        self.prec_recall = False
        self.oracle_position = False

def get_items(curr_dir, args):
    """
    This is a modified version of the SpatialAudioDataset DataLoader
    """
    with open(Path(curr_dir) / 'metadata.json') as json_file:
        json_data = json.load(json_file)
    

    num_voices = args.n_voices
    mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))

    # All voice signals
    keys = ["voice{:02}".format(i) for i in range(num_voices)]

    # Comment out this line to do voice only, no bg
    if "bg" in json_data:
        keys.append("bg")
    """
    Loading the sources
    """
    # Iterate over different sources
    all_sources = []
    target_voice_data = []
    voice_positions = []
    for key in keys:
        gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))
        assert (len(gt_audio_files) > 0)
        gt_waveforms = []

        # Iterate over different mics
        for _, gt_audio_file in enumerate(gt_audio_files):
            print(gt_audio_file)
            gt_waveform, _ = librosa.core.load(gt_audio_file,sr =  args.sr,mono=True)
            gt_waveforms.append(gt_waveform)

        single_source = np.stack(gt_waveforms)
        all_sources.append(single_source)
        
        if key == "bg":
            locs_voice = np.arctan2(json_data[key]["Position"][0][1],
                                    json_data[key]["Position"][0][0])
        else:
            locs_voice = np.arctan2(json_data[key]["Position"][1],
                                    json_data[key]["Position"][0])
        voice_positions.append(locs_voice)

    all_sources = np.stack(all_sources)  # n voices x n mics x n samples
    mixed_data = np.sum(all_sources, axis=0)  # n mics x n samples

    gt = [
        CandidateVoice(voice_positions[i], None, all_sources[i])
        for i in range(num_voices)
    ]

    return mixed_data, gt


def main(args):
    args.moving = False
    device = torch.device('cuda:1') if args.use_cuda else torch.device('cpu')

    args.device = device
    model = CosNetwork(n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint), strict=True)
    model.train = False
    model.to(device)

    all_dirs = sorted(list(Path(args.test_dir).glob('*[0-9]')))
    all_dirs = [x for x in all_dirs if check_valid_dir(x, args.n_voices)]

    if args.prec_recall and args.oracle_position:
        raise(ValueError("Either specify prec recall or oracle position"))

    if args.prec_recall:
        # True positives, false negatives, false positives
        all_tp, all_fn, all_fp = [], [], []

    else:
        # Placeholders to support multiprocessing
        all_angle_errors = [0] * len(all_dirs)
        all_input_sdr = [0] * len(all_dirs)
        all_output_sdr = [0] * len(all_dirs)

    gpu_lock = Lock()

    def evaluate_dir(idx):
        if args.debug:
            curr_writing_dir = "{:05d}".format(idx)
            if not os.path.exists(curr_writing_dir):
                os.makedirs(curr_writing_dir)
            args.writing_dir = curr_writing_dir

        curr_dir = all_dirs[idx]

        # Loads the data
        mixed_data, gt = get_items(curr_dir, args)

        # Prevents CUDA out of memory
        gpu_lock.acquire()
        if args.prec_recall:
            # Case where we don't know the number of sources
            candidate_voices = run_separation(mixed_data, model, args)

        # Case where we know the number of sources
        else:
            # Normal run
            if not args.oracle_position:
                candidate_voices = run_separation(mixed_data, model, args, 0.005)
            # In order to compute SDR or angle error, the number of outputs must match gt
            # We set a very low threshold to ensure we get the correct number of outputs
            if args.oracle_position or len(candidate_voices) < len(gt):
                print("Had to go again\n")
                candidate_voices = run_separation(mixed_data, model, args, 0.000001)

            # Use the GT positions to find the best sources
            if args.oracle_position:
                trimmed_voices = []
                for gt_idx in range(args.n_voices):
                    best_idx = np.argmin(np.array([angular_distance(x.angle,
                                               gt[gt_idx].angle) for x in candidate_voices]))
                    trimmed_voices.append(candidate_voices[best_idx])
                candidate_voices = trimmed_voices

            # Take the top N voices
            else:
                candidate_voices = candidate_voices[:args.n_voices]
            if len(candidate_voices) != len(gt):
                print(f"Not enough outputs for dir {curr_dir}. Lower threshold to evaluate.")
                return

        if args.debug:
            sf.write(os.path.join(args.writing_dir, "mixed.wav"),
                     mixed_data[0],
                     args.sr)
            for voice in candidate_voices:
                fname = "out_angle{:.2f}.wav".format(
                    voice.angle * 180 / np.pi)
                sf.write(os.path.join(args.writing_dir, fname), voice.data[0],
                         args.sr)

        gpu_lock.release()
        curr_angle_errors = []
        curr_input_sdr = []
        curr_output_sdr = []

        best_permutation, (tp, fn, fp) = find_best_permutation_prec_recall(
            [x.angle for x in gt], [x.angle for x in candidate_voices])

        if args.prec_recall:
            all_tp.append(tp)
            all_fn.append(fn)
            all_fp.append(fp)
        else:
            # Evaluate SDR and Angular Error
            for gt_idx, output_idx in enumerate(best_permutation):
                angle_error = angular_distance(candidate_voices[output_idx].angle,
                                               gt[gt_idx].angle)
                # print(angle_error)
                curr_angle_errors.append(angle_error)

                # To speed up we only evaluate channel 0. For rigorous results
                # set that to false
                input_sdr = compute_sdr(gt[gt_idx].data, mixed_data,
                                        single_channel=True)
                output_sdr = compute_sdr(gt[gt_idx].data,
                                         candidate_voices[output_idx].data, single_channel=True)
                
                curr_input_sdr.append(input_sdr)
                curr_output_sdr.append(output_sdr)

            # print(curr_input_sdr)
            # print(curr_output_sdr)

            all_angle_errors[idx] = curr_angle_errors
            all_input_sdr[idx] = curr_input_sdr
            all_output_sdr[idx] = curr_output_sdr

        #print("Running median angle error: {}".format(np.median(np.array(all_angle_errors[:idx+1])) * 180 / np.pi))
        #print("Running median SDRi: ",
        #      np.median(np.array(all_output_sdr[:idx+1]) - np.array(all_input_sdr[:idx+1])))

    for i in range(len(all_dirs)):
        evaluate_dir(i)


    # Print and save the outputs
    if args.prec_recall:
        print("Overall Precision: {} Recall: {}".format(
            sum(all_tp) / (sum(all_tp) + sum(all_fp)),
            sum(all_tp) / (sum(all_tp) + sum(all_fn))))

    else:
        print("Median Angular Error: ", np.median(np.array(all_angle_errors)) * 180 / np.pi)
        print("Median SDRi: ",
              np.median(np.array(all_output_sdr) - np.array(all_input_sdr)))
        # Uncomment to save the data for visualization
        np.save("angleerror_{}voices_{}kHz.npy".format(args.n_voices, args.sr),
                np.array(all_angle_errors).flatten())
        np.save("SDR_{}voices_{}kHz.npy".format(args.n_voices, args.sr),
                np.array([np.array(all_input_sdr).flatten(), np.array(all_output_sdr).flatten()]))


if __name__ == '__main__':
    args = Args()
    main(args)