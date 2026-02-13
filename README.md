# HMT-ECGNet  
**A Lightweight Hierarchical Multi-Lead ECG Model for PTB-XL**

---

## 1. Overview

**HMT-ECGNet** is a **lightweight hierarchical neural network** designed for 12-lead ECG classification on the **PTB-XL** dataset.  
The goal of this project is **not** to chase state-of-the-art leaderboard numbers, but to answer a more practical and honest question:

> **How much performance can we realistically extract from PTB-XL using a compact, deployment-friendly model under strict, leakage-free evaluation?**

This project demonstrates that a **~0.34M parameter hierarchical model** can match or exceed the performance of much larger CNN baselines when evaluated correctly.

---

## 2. Motivation & Problem Statement

Most modern ECG classification systems rely on:
- deep CNN stacks (ResNet-like),
- or Transformer-based architectures,
- often with **millions to tens of millions of parameters**.

While these models report strong metrics, they frequently suffer from:
- patient-level data leakage,
- improper train/validation/test separation,
- threshold tuning on the test set,
- over-reliance on accuracy despite heavy class imbalance.

### This project focuses on:
- **strict adherence to official PTB-XL splits**,  
- **no test-time tuning**,  
- **transparent reporting of AUROC and F1**,  
- and **parameter efficiency**.

---

## 3. Dataset: PTB-XL

- **Records:** ~21,800 ECGs  
- **Leads:** 12  
- **Duration:** 10 seconds  
- **Sampling rate:** 500 Hz (downsampled to 250 Hz)  
- **Splits:** Official PTB-XL train / validation / test folds  

### Tasks

- **Multi-label classification (5 diagnostic superclasses)**  
  - NORM, MI, STTC, CD, HYP  

- **Binary classification**
  - MI vs Normal  
  - Normal vs Abnormal  

📌 **Important:**  
All experiments use the **official PTB-XL splits**.  
There is **no patient leakage**, **no test-set threshold tuning**, and **no post-hoc metric adjustment**.

---

## 4. Architecture: HMT-ECGNet

### High-Level Design
12-Lead ECG (2200 samples)
│-- Shared Per-Lead Temporal Encoder
│-- Lead-wise Feature Tokens
│-- Hierarchical Cross-Lead Aggregation
│-- Global Representation
│-- Classification Head



### Design Principles

- **Hierarchical modeling**
  - Temporal patterns are learned per lead
  - Cross-lead relationships are learned at higher levels
- **Weight sharing across leads** for efficiency
- **No heavy self-attention or Transformers**
- **Strong inductive bias for ECG structure**

**Total parameters:** ~**338K**

---

## 5. Training Protocol

- **Optimizer:** AdamW  
- **Learning rate schedule:** Cosine Annealing  
- **Loss:**
  - Multi-label: `AsymmetricFocalLoss` with class imbalance handling  
  - Binary: `BCEWithLogitsLoss`  
- **Regularization:**
  - Early stopping
  - Mild data augmentation
- **Reproducibility:**
  - Fixed random seeds
  - Deterministic data splits

---

## 6. Results

### Multi-Label Classification (5 Classes, Test Set)

| Model | Params | AUROC (macro) | F1 (macro) |
|-----|-------:|--------------:|-----------:|
| **HMT-ECGNet** | **0.34M** | **0.9206** | **~0.73** |
| ResNet-1D Baseline | ~8.7M | ~0.90 | ~0.70 |

📌 **Key insight:**  
Despite being **~25× smaller**, HMT-ECGNet matches or exceeds the ResNet baseline under identical evaluation conditions.

---

### Binary Classification — MI vs Normal (Test Set)

| Model | Params | AUROC | F1 | Accuracy |
|-----|-------:|------:|---:|---------:|
| **HMT-ECGNet** | **0.34M** | **~0.98** | **~0.89** | ~0.92 |
| ResNet-1D Baseline | ~8.7M | ~0.97 | ~0.87 | ~0.92 |

📌 **Observation:**  
Accuracy saturates due to ambiguous ECGs, while AUROC remains high, indicating strong class separability.

---

## 7. Why Performance Saturates on PTB-XL

Even strong models plateau because:

- ECG labels (especially MI) have **inter-observer disagreement**
- Borderline and chronic cases blur class boundaries
- PTB-XL diagnostic labels are **not pixel-perfect ground truth**

This explains why:
- AUROC can be very high,
- but accuracy and F1 improve slowly beyond a point.

---

## 8. Ensemble & Error Analysis

- Multi-seed training and ensemble averaging were evaluated
- Ensembles improved stability but **did not significantly improve accuracy**
- Remaining errors are **systematic and data-driven**, not due to variance
- Confirms a **realistic performance ceiling** on PTB-XL

---

## 9. Project Structure

hmt_ecgnet/
├── artifacts/
│ └── binary_threshold.json
│ └── loss_graph.png
│ └── mi_best.pth
│ └── multilabel_best.pth
│ └── multilabel_thresholds.json
│ └── resnet_baseline.pth
├── models/
│ └── hmt_ecgnet.py
│ └── resnet1d.py
├── api.py
├── app.py
├── config.py
├── dataset.py
├── eval_multilabel.py
├── eval_binary.py
├── requirement.txt
├── threshold_search_multilabel.py
├── threshold_search.py
├── train_binary.py
├── train_multilabel.py
├── train_resnet_baseline.py
├── transforms.py
└── README.md


---

## 10. Comparison Summary

| Aspect | HMT-ECGNet | ResNet-1D |
|-----|-----------|----------|
| Parameters | **~0.34M** | ~8.7M |
| Architecture | Hierarchical | Deep CNN |
| Attention | ❌ | ❌ |
| Deployment-friendly | ✅ | ❌ |
| AUROC (ML) | **~0.92** | ~0.90 |
| F1 (ML) | **~0.73** | ~0.70 |

---

## 11. References

1. Wagner, P. et al.  
   **PTB-XL: A Large Publicly Available Electrocardiography Dataset**  
   *PhysioNet, 2020.*

2. Ribeiro, A. H. et al.  
   **Automatic Diagnosis of the 12-Lead ECG Using Deep Neural Networks**  
   *Nature Communications, 2020.*

3. Hannun, A. et al.  
   **Cardiologist-Level Arrhythmia Detection with Deep Neural Networks**  
   *Nature Medicine, 2019.*

4. Rajpurkar, P. et al.  
   **Cardiologist-Level Arrhythmia Detection Using a Deep Neural Network**  
   *arXiv:1707.01836.*

5. Yao, Q. et al.  
   **Time-Invariant Representation Learning for ECG Classification**  
   *IEEE Transactions on Biomedical Engineering, 2020.*

---

## 12. Disclaimer

This project follows **strict evaluation protocols**.  
All reported results are obtained **without data leakage**, **without test-set tuning**, and **without metric cherry-picking**.

---

## 13. Author Note

This work was developed as a **research-oriented engineering project**, emphasizing:

- efficiency over scale,
- reproducibility over inflated metrics,
- and honest analysis over paper-style optimization.

The goal is to demonstrate **what is realistically achievable** with well-designed lightweight models on real ECG data.