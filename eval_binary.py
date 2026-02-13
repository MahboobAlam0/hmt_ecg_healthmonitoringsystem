import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from config import BATCH_SIZE, N_LEADS

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--threshold", type=float, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA ----------------
test_ds = PTBXLDataset(split="test", task="binary", binary_task=args.task)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = HMT_ECGNet(num_classes=1, num_leads=N_LEADS).to(device)
ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------------- INFERENCE ----------------
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

# ---------------- THRESHOLD ----------------
if args.threshold is None:
    threshold = 0.5
else:
    threshold = args.threshold

preds = (probs >= threshold).astype(int)

# ---------------- METRICS ----------------
f1 = f1_score(labels, preds)
auroc = roc_auc_score(labels, probs)
acc = accuracy_score(labels, preds)

print("\n=== BINARY TEST RESULTS (MI vs NORMAL) ===")
print(f"Threshold : {threshold}")
print(f"AUROC     : {auroc:.4f}")
print(f"F1        : {f1:.4f}")
print(f"Accuracy  : {acc:.4f}")