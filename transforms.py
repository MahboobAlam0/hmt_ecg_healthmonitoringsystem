# transforms.py
import numpy as np
from scipy.signal import butter, filtfilt, resample

from config import FS_ORIG, FS_TARGET, TARGET_LEN, N_LEADS


def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    """
    Band-pass filter: remove baseline wander and high-frequency noise.

    sig: [n_leads, T] numpy array
    """
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")

    # apply per lead
    filtered = np.zeros_like(sig)
    for i in range(sig.shape[0]):
        filtered[i] = filtfilt(b, a, sig[i])
    return filtered


def preprocess_signal(sig, fs_orig=FS_ORIG, fs_target=FS_TARGET):
    """
    Correct PTB-XL preprocessing:
    - band-pass filter 0.5–40 Hz
    - resample to fs_target
    - CENTER crop to TARGET_LEN
    - per-lead z-score
    """

    # band-pass
    sig = bandpass_filter(sig, fs_orig)

    # resample
    T_orig = sig.shape[1]
    target_len = int(round(T_orig * fs_target / fs_orig))
    sig = resample(sig, target_len, axis=1)
    T = sig.shape[1]

    if T >= TARGET_LEN:
        start = T // 2 - TARGET_LEN // 2
        sig = sig[:, start:start + TARGET_LEN]
    else:
        pad = TARGET_LEN - T
        left = pad // 2
        right = pad - left
        sig = np.pad(sig, ((0, 0), (left, right)), mode="constant")

    # normalize per lead AFTER cropping
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-6
    sig = (sig - mean) / std

    return sig.astype(np.float32)



class ECGAugmentation:
    def __init__(self):
        self.noise_std = 0.05
        self.scale_low = 0.8
        self.scale_high = 1.2
        self.max_time_shift = 150
        self.lead_dropout_prob = 0.5

    def __call__(self, sig):
        # time shift
        shift = np.random.randint(-self.max_time_shift, self.max_time_shift)
        sig = np.roll(sig, shift, axis=1)

        # lead dropout
        if np.random.rand() < self.lead_dropout_prob:
            leads = np.random.choice(np.arange(sig.shape[0]), 3, replace=False)
            sig[leads] = 0

        # scale
        scale = np.random.uniform(self.scale_low, self.scale_high)
        sig = sig * scale

        # noise
        sig += np.random.normal(0, self.noise_std, sig.shape)

        return sig.astype(np.float32)