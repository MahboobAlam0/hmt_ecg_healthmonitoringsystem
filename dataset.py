# dataset.py
import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample

from config import PTBXL_ROOT, DIAG_SUPERCLASSES

FS_ORIG = 500
FS_TARGET = 100
TARGET_LEN = FS_TARGET * 10


def preprocess_signal(sig):
    """
    Training preprocessing ONLY
    sig: (12, T)
    """
    nyq = 0.5 * FS_ORIG
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")

    for i in range(sig.shape[0]):
        sig[i] = filtfilt(b, a, sig[i])

    T_new = int(sig.shape[1] * FS_TARGET / FS_ORIG)
    sig = resample(sig, T_new, axis=1)

    # Z-score per lead (TRAINING ONLY)
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (
        sig.std(axis=1, keepdims=True) + 1e-6
    )

    if sig.shape[1] > TARGET_LEN:
        sig = sig[:, :TARGET_LEN]
    else:
        pad = TARGET_LEN - sig.shape[1]
        sig = np.pad(sig, ((0, 0), (0, pad)))

    return sig.astype(np.float32)


class PTBXLDataset(Dataset):
    def __init__(self, split="train", task="multilabel", binary_task=None):
        self.task = task
        self.binary_task = binary_task

        df = pd.read_csv(os.path.join(PTBXL_ROOT, "ptbxl_database.csv"))
        scp = pd.read_csv(os.path.join(PTBXL_ROOT, "scp_statements.csv"))

        scp = scp.rename(columns={"Unnamed: 0": "scp_code"}).set_index("scp_code")
        scp = scp[scp["diagnostic"] == 1]
        self.code_map = scp["diagnostic_class"].to_dict()

        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
        df["labels"] = df["scp_codes"].apply(self.map_to_superclasses)

        folds = {"train": [1,2,3,4,5,6,7,8], "val": [9], "test": [10]}[split]
        self.records = df[df["strat_fold"].isin(folds)].reset_index(drop=True)

        print(f"{split.upper()} set: {len(self.records)} samples")

    def map_to_superclasses(self, scp_dict):
        return {
            self.code_map[c] for c in scp_dict if c in self.code_map
        }

    def load_ecg(self, filename):
        sig, _ = wfdb.rdsamp(os.path.join(PTBXL_ROOT, filename))
        return preprocess_signal(sig.T)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        x = self.load_ecg(row["filename_hr"])

        if self.task == "multilabel":
            y = np.zeros(len(DIAG_SUPERCLASSES), dtype=np.float32)
            for i, cls in enumerate(DIAG_SUPERCLASSES):
                if cls in row["labels"]:
                    y[i] = 1.0
        else:
            raise NotImplementedError

        return torch.tensor(x), torch.tensor(y)