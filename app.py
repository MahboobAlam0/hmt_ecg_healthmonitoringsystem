# app.py
import os
import time
import numpy as np
import wfdb
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample

from config import PTBXL_ROOT

# CONFIG
API_URL = "http://127.0.0.1:8000/predict"

LEAD_NAMES = [
    "I","II","III","aVR","aVL","aVF",
    "V1","V2","V3","V4","V5","V6"
]

FS_ORIG = 500
FS_TARGET = 100
DURATION_SEC = 10
TARGET_LEN = FS_TARGET * DURATION_SEC

# PAGE SETUP
st.set_page_config(
    page_title="ECG AI Diagnostic System",
    layout="wide",
)

st.title("12-Lead ECG AI Diagnostic System")
st.markdown(
    "**Live demo on unseen PTB-XL TEST ECGs**  \n"
    "Lightweight hierarchical model • No data leakage • Realistic evaluation"
)

# DATA LOADING (CACHED)
@st.cache_data(show_spinner=False)
def load_test_metadata():
    df = pd.read_csv(os.path.join(PTBXL_ROOT, "ptbxl_database.csv"))
    return df[df["strat_fold"] == 10].reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_and_preprocess_ecg(filename):
    sig, _ = wfdb.rdsamp(os.path.join(PTBXL_ROOT, filename))
    sig = sig.T  # (12, T)

    # Bandpass filter
    nyq = 0.5 * FS_ORIG
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
    for i in range(12):
        sig[i] = filtfilt(b, a, sig[i])

    # Resample
    sig = resample(sig, TARGET_LEN, axis=1)

    return sig.astype(np.float32)

df_test = load_test_metadata()

# ECG PLOTTING
def plot_ecg_frame(ecg, end_idx):
    fig, axes = plt.subplots(12, 1, figsize=(24, 14), sharex=True)

    x = np.arange(end_idx)

    for i in range(12):
        axes[i].plot(x, ecg[i, :end_idx], lw=1.1)
        axes[i].set_ylabel(
            LEAD_NAMES[i],
            rotation=0,
            labelpad=28,
            fontsize=10
        )
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (compressed)")
    plt.tight_layout()
    return fig

# SIDEBAR CONTROLS
st.sidebar.header("ECG Sample Selector")

sample_idx = st.sidebar.slider(
    "Select ECG from TEST set",
    0,
    len(df_test) - 1,
    0
)

row = df_test.iloc[sample_idx]
ecg = load_and_preprocess_ecg(row["filename_hr"])

# ECG ANIMATION (AUTO PLAY)
st.subheader(f"ECG Sample #{sample_idx}")

plot_placeholder = st.empty()

for t in range(50, TARGET_LEN + 1, 25):
    fig = plot_ecg_frame(ecg, t)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
    time.sleep(0.025)

# AI INFERENCE (ONCE)
st.subheader("AI Diagnosis")

payload = {
    f"lead{i+1}": ecg[i].tolist()
    for i in range(12)
}

try:
    response = requests.post(API_URL, json=payload, timeout=15)
    response.raise_for_status()
    result = response.json()

except Exception:
    st.error(
        "AI server not reachable. "
        "Make sure FastAPI is running on port 8000."
    )
    st.stop()

# RESULTS DISPLAY
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
    st.error("High likelihood of Myocardial Infarction")
else:
    st.success("No strong MI indication")