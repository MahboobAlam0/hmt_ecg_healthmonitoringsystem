# eval_with_thresh.py
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from dataset import PTBXLDiagnosticDataset
from models import HMT_ECGNet
from config import N_LEADS, BATCH_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model
    model = HMT_ECGNet(num_classes=1, num_leads=N_LEADS).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # load test dataset
    test_ds = PTBXLDiagnosticDataset(
        split="test", task="mi_vs_norm", use_augmentation=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x).view(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)

    preds = (probs >= args.threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    print("\nTest results (MI vs NORM):")
    print(f"  Threshold: {args.threshold:.3f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()