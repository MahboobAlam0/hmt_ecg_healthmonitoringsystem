import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    hamming_loss,
)

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from config import BATCH_SIZE, N_LEADS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--thresholds", type=str, default="multilabel_thresholds.json")
args = parser.parse_args()


@torch.no_grad()
def main():
    # ---------------- Dataset ----------------
    test_ds = PTBXLDataset(split="test", task="multilabel")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ---------------- Model ----------------
    model = HMT_ECGNet(num_classes=5, num_leads=N_LEADS).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---------------- Thresholds ----------------
    with open(args.thresholds, "r") as f:
        thresholds = json.load(f)
    thresh = np.array([thresholds[c] for c in CLASSES])

    print("Using per-class thresholds:")
    for c, t in zip(CLASSES, thresh):
        print(f"  {c}: {t:.2f}")

    # ---------------- Inference ----------------
    all_probs, all_targets = [], []

    for x, y in test_loader:
        x = x.to(DEVICE)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)
    preds = (probs >= thresh).astype(int)

    # ---------------- Metrics ----------------
    auroc_macro = roc_auc_score(targets, probs, average="macro")
    auprc_macro = average_precision_score(targets, probs, average="macro")
    f1_macro = f1_score(targets, preds, average="macro")
    f1_micro = f1_score(targets, preds, average="micro")

    exact_match = accuracy_score(targets, preds)        # Subset accuracy
    h_loss = hamming_loss(targets, preds)

    print("\n=== MULTI-LABEL TEST RESULTS (Threshold-Optimized) ===")
    print(f"AUROC_macro     : {auroc_macro:.4f}")
    print(f"AUPRC_macro     : {auprc_macro:.4f}")
    print(f"F1_macro        : {f1_macro:.4f}")
    print(f"F1_micro        : {f1_micro:.4f}")
    print(f"Exact Match     : {exact_match:.4f}")
    print(f"Hamming Loss    : {h_loss:.4f}")

    print("\nPer-class F1:")
    for i, cls in enumerate(CLASSES):
        f1 = f1_score(targets[:, i], preds[:, i])
        print(f"  {cls:5s}: {f1:.4f}")


if __name__ == "__main__":
    main()