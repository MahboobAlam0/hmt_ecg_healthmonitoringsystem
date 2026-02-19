# config.py
import os

# PROJECT ROOT
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# PTB-XL PATHS
PTBXL_ROOT = os.path.join(PROJECT_ROOT, "EcgDataset")

PTBXL_DB_CSV = os.path.join(PTBXL_ROOT, "ptbxl_database.csv")
PTBXL_SCP_CSV = os.path.join(PTBXL_ROOT, "scp_statements.csv")

# LABELS (official PTB-XL diagnostic superclasses)
DIAG_SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

# ECG CONSTANTS
N_LEADS = 12
FS_ORIG = 250
FS_TARGET = 10
TARGET_LEN = FS_ORIG*FS_TARGET

# TRAINING
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 60

# LOGGING / CHECKPOINTS
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOG_DIR = os.path.join(PROJECT_ROOT, "artifacts")