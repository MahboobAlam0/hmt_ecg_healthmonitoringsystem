import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from config import BATCH_SIZE, N_LEADS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


CKPT_PATH = r"D:\Health Monitoring System\artifacts\multilabel_best.pth"


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print("Using checkpoint:", CKPT_PATH)

    # Dataset (VAL split)
    val_ds = PTBXLDataset(split="val", task="multilabel")
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = HMT_ECGNet(num_classes=5, num_leads=N_LEADS).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect outputs
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    # Threshold search
    thresholds = {}

    print("\nSearching optimal thresholds per class...\n")

    for i, cls in enumerate(CLASSES):
        best_f1, best_t = 0, 0.5

        for t in np.linspace(0.05, 0.95, 181):  # finer search
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(targets[:, i], preds)

            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[cls] = float(best_t)
        print(f"{cls}: threshold = {best_t:.3f}, F1 = {best_f1:.4f}")

    # Save thresholds to artifacts
    save_path = r"D:\Health Monitoring System\artifacts\multilabel_thresholds.json"

    with open(save_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nSaved thresholds to: {save_path}")


if __name__ == "__main__":
    main()