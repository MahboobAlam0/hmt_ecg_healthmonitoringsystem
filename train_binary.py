import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from utils_seed import set_seed
from config import BATCH_SIZE, LR, WEIGHT_DECAY, N_EPOCHS, N_LEADS


# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["mi_vs_norm", "norm_vs_abnormal"], required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=N_EPOCHS)
args = parser.parse_args()


def get_loader(ds, shuffle):
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda
    )


def main():
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"hmt_binary_{args.task}_seed{args.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- DATA ----------------
    train_ds = PTBXLDataset(split="train", task="binary", binary_task=args.task)
    val_ds   = PTBXLDataset(split="val",   task="binary", binary_task=args.task)

    train_loader = get_loader(train_ds, shuffle=True)
    val_loader   = get_loader(val_ds, shuffle=False)

    # ---------------- MODEL ----------------
    model = HMT_ECGNet(num_classes=1, num_leads=N_LEADS).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]"):
            x, y = x.to(device), y.to(device).float()

            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATE --------
        model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x).squeeze(1)
                all_logits.append(logits.cpu())
                all_targets.append(y)

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        probs = torch.sigmoid(logits).numpy()
        val_auc = roc_auc_score(targets.numpy(), probs)
        val_f1  = f1_score(targets.numpy(), probs >= 0.5)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"AUROC={val_auc:.4f} | F1={val_f1:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(log_dir, f"hmt_binary_{args.task}_best.pth"),
            )
            print("  -> Best model saved")

    print("Training complete.")


if __name__ == "__main__":
    main()