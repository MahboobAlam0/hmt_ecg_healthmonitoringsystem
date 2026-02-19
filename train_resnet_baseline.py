import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from dataset import PTBXLDataset
from config import BATCH_SIZE
from models.resnet1d import ResNet1D  


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data
train_ds = PTBXLDataset(split="train", task="multilabel")
val_ds = PTBXLDataset(split="val", task="multilabel")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Model
model = ResNet1D(num_classes=5).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Validation function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(targets, probs, average="macro")
    f1 = f1_score(targets, preds, average="macro")

    return auroc, f1


# Early stopping setup
PATIENCE = 6
best_f1 = 0.0
epochs_no_improve = 0
best_state = None


# Training loop
for epoch in range(1, 61):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    val_auroc, val_f1 = evaluate(model, val_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"loss={train_loss:.4f} | "
        f"AUROC_macro={val_auroc:.4f} | "
        f"F1_macro={val_f1:.4f}"
    )

    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        epochs_no_improve = 0
        best_state = copy.deepcopy(model.state_dict())
        print("  -> Best model saved (by F1)")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break


# Save best model
print("\nSaving best ResNet baseline model...")
torch.save(
    {"model_state_dict": best_state},
    "resnet_baseline_best.pth",
)

print("Training finished.")
