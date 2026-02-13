import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from config import BATCH_SIZE, N_LEADS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mi_vs_norm", "norm_vs_abnormal"], required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    # ---------------- Dataset ----------------
    val_ds = PTBXLDataset(split="val", task="binary", binary_task=args.task)

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    # ---------------- Model ----------------
    model = HMT_ECGNet(num_classes=1, num_leads=N_LEADS).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---------------- Collect outputs ----------------
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(y.numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    # ---------------- Threshold search ----------------
    best_f1, best_t = 0, 0.5

    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest threshold: {best_t:.2f}")
    print(f"Best F1 score : {best_f1:.4f}")


if __name__ == "__main__":
    main()