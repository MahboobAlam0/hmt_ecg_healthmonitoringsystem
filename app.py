# app.py — Self-contained Streamlit ECG Diagnostic App (HF Spaces compatible)
import os
import json
import numpy as np
import torch
import wfdb
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample

# Local imports
from models.hmt_ecgnet import HMT_ECGNet
from transforms import preprocess_signal
from config import N_LEADS

# Constants
DIAG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI_BINARY_THRESHOLD = 0.05

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

FS_ORIG = 500
FS_TARGET = 100
DURATION_SEC = 10
TARGET_LEN = FS_TARGET * DURATION_SEC

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EcgDataset")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# Page Config 
st.set_page_config(
    page_title="ECG AI Diagnostic System",
    page_icon="",
    layout="wide",
)


# MODEL LOADING (cached — runs once)

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HMT_ECGNet(num_classes=5, num_leads=N_LEADS).to(device)

    ckpt_path = os.path.join(ARTIFACTS_DIR, "multilabel_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, device


@st.cache_data(show_spinner=False)
def load_thresholds():
    path = os.path.join(ARTIFACTS_DIR, "multilabel_thresholds.json")
    with open(path) as f:
        return json.load(f)


#  DATA DOWNLOADING & LOADING

@st.cache_data(show_spinner="Downloading PTB-XL sample data...")
def download_ptbxl_data():
    """Download PTB-XL database CSV + a subset of high-res records."""
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        # Download the metadata CSV and SCP statements
        wfdb.dl_database("ptb-xl", dl_dir=DATA_DIR, records="all", annotators=None)

    return True


@st.cache_data(show_spinner=False)
def load_test_metadata():
    csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    df_test = df[df["strat_fold"] == 10].reset_index(drop=True)
    return df_test


@st.cache_data(show_spinner=False)
def load_and_preprocess_ecg(filename):
    """Load a single ECG record from PTB-XL and preprocess for display."""
    filepath = os.path.join(DATA_DIR, filename)

    # Download this specific record if not present
    record_dir = os.path.dirname(filepath)
    os.makedirs(record_dir, exist_ok=True)

    if not os.path.exists(filepath + ".hea"):
        # Download just this record
        rel_path = filename.replace("\\", "/")
        try:
            wfdb.dl_database(
                "ptb-xl",
                dl_dir=DATA_DIR,
                records=[rel_path],
            )
        except Exception:
            st.error(f"Could not download record: {filename}")
            return None

    sig, _ = wfdb.rdsamp(filepath)
    sig = sig.T  # (12, T)

    # Bandpass filter for display
    nyq = 0.5 * FS_ORIG
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
    for i in range(12):
        sig[i] = filtfilt(b, a, sig[i])

    # Resample for display
    sig = resample(sig, TARGET_LEN, axis=1)

    return sig.astype(np.float32)


#  INFERENCE (runs directly — no FastAPI needed)

def run_inference(ecg_display, model, device, thresholds):
    """Run model inference on preprocessed ECG data."""
    # Re-preprocess from display signal for model input
    ecg_for_model = preprocess_signal(ecg_display.copy())

    x = torch.tensor(ecg_for_model, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]

    result = {}
    predicted = []

    for cls, p in zip(DIAG_CLASSES, probs):
        thr = thresholds[cls]
        result[cls] = float(p)
        if p >= thr:
            predicted.append(cls)

    mi_prob = float(probs[1])

    return {
        "probabilities": result,
        "predicted_classes": predicted,
        "mi_probability": mi_prob,
        "mi_risk": mi_prob >= MI_BINARY_THRESHOLD,
    }


#  ECG PLOTTING

def plot_ecg(ecg):
    """Plot full 12-lead ECG as a static figure."""
    fig, axes = plt.subplots(12, 1, figsize=(24, 14), sharex=True)

    x = np.arange(ecg.shape[1])

    for i in range(12):
        axes[i].plot(x, ecg[i], lw=1.1, color="#1f77b4")
        axes[i].set_ylabel(
            LEAD_NAMES[i],
            rotation=0,
            labelpad=28,
            fontsize=10,
        )
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    return fig


#  MAIN APP

st.title(" 12-Lead ECG AI Diagnostic System")
st.markdown(
    "**Live demo on unseen PTB-XL TEST ECGs**  \n"
    "Lightweight hierarchical model • No data leakage • Realistic evaluation"
)

# Load model & data 
model, device = load_model()
thresholds = load_thresholds()

with st.spinner("Preparing PTB-XL test data..."):
    download_ptbxl_data()
    df_test = load_test_metadata()

if len(df_test) == 0:
    st.error("No test data found. Please check the PTB-XL dataset.")
    st.stop()

# Sidebar 
st.sidebar.header("ECG Sample Selector")

sample_idx = st.sidebar.slider(
    "Select ECG from TEST set",
    0,
    len(df_test) - 1,
    0,
)

row = df_test.iloc[sample_idx]
ecg = load_and_preprocess_ecg(row["filename_hr"])

if ecg is None:
    st.warning("Could not load this ECG record. Try another sample.")
    st.stop()

# ECG Display 
st.subheader(f"ECG Sample #{sample_idx}")

fig = plot_ecg(ecg)
st.pyplot(fig)
plt.close(fig)

# AI Inference 
st.subheader("AI Diagnosis")

with st.spinner("Running inference..."):
    result = run_inference(ecg, model, device, thresholds)

# Results 
st.markdown("### Per-class Probabilities")

cols = st.columns(5)
for col, (cls, prob) in zip(cols, result["probabilities"].items()):
    col.metric(cls, f"{prob:.3f}")

st.markdown("### Final Predicted Classes")

if result["predicted_classes"]:
    st.error(", ".join(result["predicted_classes"]))
else:
    st.success("Normal ECG — No pathology detected")

st.markdown("### Myocardial Infarction Screening")

st.metric("MI Probability", f"{result['mi_probability']:.3f}")

if result["mi_risk"]:
    st.error("⚠️ High likelihood of Myocardial Infarction")
else:
    st.success(" No strong MI indication")

# Footer 
st.markdown("---")
st.caption(
    "⚕️ **Disclaimer:** This system is for research and demonstration only. "
    "Not intended for clinical diagnosis or treatment."
)