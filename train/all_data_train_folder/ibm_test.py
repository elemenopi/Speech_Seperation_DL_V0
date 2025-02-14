import argparse
import json
import multiprocessing.dummy as mp
import os
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import tqdm

from scipy.signal import stft, istft

from eval_utils import compute_sdr
from eval import get_items
from utils import check_valid_dir


def compute_ibm(gt, mix, alpha, theta=0.5):
    """
    Computes the Ideal Binary Mask SI-SDR
    gt: (n_voices, n_channels, t)
    mix: (n_channels, t)
    """
    n_voices = gt.shape[0]
    print(f"n_voices {n_voices}")
    nfft = 2048
    eps = np.finfo(np.float64).eps
    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape # (6, nfft//2 +1, n_frame)

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # compute STFT of target source
        print(gt[gt_idx])
        Yj = stft(gt[gt_idx], nperseg=nfft)[2]

        # Create binary Mask
        mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X) ** alpha))
        mask[np.where(mask >= theta)] = 1
        mask[np.where(mask < theta)] = 0

        Yj = np.multiply(X, mask)
        target_estimate = istft(Yj)[1][:,:N]
        print(target_estimate.shape)
        # Save each separated channel individually
        for ch in range(target_estimate.shape[0]):
            output_file_path = f"separated_source_{gt_idx}_channel_{ch}.wav"
            print(target_estimate[ch].shape)
            sf.write(output_file_path, target_estimate[ch].astype(np.float32), 44100)
        
        estimates.append(target_estimate)

    estimates = np.array(estimates) # (nvoice, 6, 6*sr)

    # eval
    eval_mix = np.repeat(mix[np.newaxis, :, :], n_voices, axis=0) # (nvoice, 6, 6*sr)
    eval_gt = gt # (nvoice, 6, 6*sr)
    eval_est = estimates

    SDR_in = []
    SDR_out = []
    for i in range(n_voices):
        SDR_in.append(compute_sdr(eval_gt[i], eval_mix[i], single_channel=True)) # scalar
        SDR_out.append(compute_sdr(eval_gt[i], eval_est[i], single_channel=True)) # scalar

    output = np.array([SDR_in, SDR_out]) # (2, nvoice)

    return output


def main(args):
    all_dirs = sorted(list(Path(args.input_dir).glob('*[0-9]')))
    all_dirs = [x for x in all_dirs if check_valid_dir(x, args.n_voices)]

    all_input_sdr = [0] * len(all_dirs)
    all_output_sdr = [0] * len(all_dirs)

    def evaluate_dir(idx):
        curr_dir = all_dirs[idx]
        # Loads the data
        mixed_data, gt = get_items(curr_dir, args)
        gt = np.array([x.data for x in gt])
        output = compute_ibm(gt, mixed_data, alpha=args.alpha)
        all_input_sdr[idx] = output[0]
        all_output_sdr[idx] = output[1]
        print("Running median SDRi: ",
              np.median(np.array(all_output_sdr[:idx+1]) - np.array(all_input_sdr[:idx+1])))

    evaluate_dir(1)
    
    # tqdm.tqdm(pool.imap(evaluate_dir, range(len(all_dirs))), total=len(all_dirs))

    print("Median SI-SDRi: ",
          np.median(np.array(all_output_sdr).flatten() - np.array(all_input_sdr).flatten()))

    np.save("IBM_{}voices_{}kHz.npy".format(args.n_voices, args.sr),
            np.array([np.array(all_input_sdr).flatten(), np.array(all_output_sdr).flatten()]))




class Args:
    def __init__(self):
        self.input_dir = '/app/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/eval_sdr_simple_0610'  # Replace with actual path

        self.sr = 44100
        self.n_channels = 6
        self.n_workers = 8
        self.n_voices = 2
        self.alpha = 1 #tocheck




if __name__ == '__main__':
    args = Args()
    main(args)