# train_multilabel.py

import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from dataset import PTBXLDataset
from models.hmt_ecgnet import HMT_ECGNet
from utils_seed import set_seed
from config import BATCH_SIZE, LR, WEIGHT_DECAY, N_EPOCHS, N_LEADS


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=N_EPOCHS)
parser.add_argument("--patience", type=int, default=8)
args = parser.parse_args()


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------
def get_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ------------------------------------------------------------
# Compute class imbalance weights
# ------------------------------------------------------------
def compute_pos_weights(dataset):
    print("Computing class imbalance weights...")

    labels = []
    with torch.no_grad():
        for i in range(len(dataset)):
            labels.append(dataset[i][1])

    labels = torch.stack(labels)
    pos = labels.sum(dim=0)
    neg = labels.size(0) - pos

    weights = (neg / (pos + 1e-6)).float()
    print("Pos weights:", weights)

    return weights


# ------------------------------------------------------------
# Asymmetric Focal Loss
# ------------------------------------------------------------
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2, gamma_neg=1, pos_weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)
        gamma = torch.where(
            targets == 1,
            self.gamma_pos,
            self.gamma_neg
        )

        loss = (1 - pt) ** gamma * bce
        return loss.mean()


# ------------------------------------------------------------
# Plot Curves
# ------------------------------------------------------------
def plot_curves(history, save_dir):

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["loss"])
    plt.title("Train Loss")
    plt.xlabel("Epoch")

    # AUROC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_auc"])
    plt.title("Validation AUROC")
    plt.xlabel("Epoch")

    # F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_f1"])
    plt.title("Validation F1_macro")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_name = f"hmt_multilabel_{datetime.now():%Y%m%d-%H%M%S}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Dataset ----------------
    train_ds = PTBXLDataset("train", "multilabel")
    val_ds   = PTBXLDataset("val",   "multilabel")

    train_loader = get_loader(train_ds, True)
    val_loader   = get_loader(val_ds, False)

    # ---------------- Model ----------------
    model = HMT_ECGNet(num_classes=5, num_leads=N_LEADS).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---------------- Loss ----------------
    pos_weights = compute_pos_weights(train_ds).to(device)

    criterion = AsymmetricFocalLoss(
        gamma_pos=2,
        gamma_neg=1,
        pos_weight=pos_weights,
    )

    # ---------------- Optimizer ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # ---------------- Early Stopping (AUROC) ----------------
    best_auc = 0.0
    patience_counter = 0

    history = {
        "loss": [],
        "val_auc": [],
        "val_f1": [],
    }

    # ==================================================
    # Training Loop
    # ==================================================
    for epoch in range(1, args.epochs + 1):

        # ===== Train =====
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ===== Validation =====
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_targets.append(y)

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        preds = (probs >= 0.5).astype(int)

        val_auc = roc_auc_score(targets, probs, average="macro")
        val_f1 = f1_score(targets, preds, average="macro")

        history["loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"AUROC_macro={val_auc:.4f} | "
            f"F1_macro={val_f1:.4f}"
        )

        # ===== Early Stopping on AUROC =====
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(log_dir, "hmt_multilabel_best.pth"),
            )

            print("  -> Best model saved (by AUROC)")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Plot curves
    plot_curves(history, log_dir)

    print("Training finished.")


if __name__ == "__main__":
    main()