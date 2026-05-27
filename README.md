# Gesture-Controlled Industrial Automation

**Hand Gesture Recognition for Industrial Robot Control using IMU Sensors, CNN-LSTM, and XAI**

A full machine learning pipeline for recognizing 11 hand gestures from 6-axis IMU (accelerometer + gyroscope) data and mapping them to safety-critical industrial robot commands. Trained and evaluated on the [HGAG Dataset](https://www.kaggle.com/datasets/hlfbldprnc/hgag-data) using Leave-One-Subject-Out (LOSO) cross-validation.

---

## 📋 Project Overview

This project implements a production-ready gesture recognition system with:

- **Dataset:** HGAG-DATA1 — 43 subjects × 11 gestures × 6 IMU channels @ 200 Hz
- **Architecture:** CNN-LSTM (3× Conv1D → 2× MaxPool → BatchNorm → LSTM → Dense)
- **Evaluation Protocol:** Leakage-safe LOSO cross-validation (subject-pure train/val/test splits)
- **XAI:** 1D Grad-CAM for temporal saliency attribution per gesture class
- **Safety Layer:** Hungarian algorithm for optimal gesture-to-command assignment by error risk
- **Deployment:** TFLite dynamic + INT8 quantization, confidence gating @ 0.70, <100 ms latency budget

---

## 🗂️ File Structure

```
gesture-controlled-industrial-automation/
│
├── hgag_cnn_lstm_final.ipynb      ← Main Kaggle notebook (single-cell, GPU-ready)
│
├── README.md                       ← This file
├── ARCHITECTURE.md                 ← CNN-LSTM design rationale & layer-by-layer explanation
├── DATASET.md                      ← Dataset structure, preprocessing audit, signal notes
├── RESULTS.md                      ← Evaluation metrics, confusion analysis, XAI summary
├── DEPLOYMENT.md                   ← TFLite export, confidence gating, API contract, latency
│
├── requirements.txt                ← Python dependencies
└── .gitignore                      ← Ignores large outputs, checkpoints, dataset files
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- GPU recommended (Kaggle / Colab with T4/P100)
- HGAG dataset from Kaggle: `hlfbldprnc/hgag-data`

### Running on Kaggle

1. Create a new Kaggle Notebook (GPU accelerator ON)
2. Add the dataset: `hlfbldprnc/hgag-data`
3. Upload `hgag_cnn_lstm_final.ipynb` or paste its contents into a single code cell
4. Click **Run All**

All outputs are written to `/kaggle/working/hgag_results_final/`.

### Local Setup

```bash
git clone https://github.com/niloysharma2005-pixel/gesture-controlled-industrial-automation
cd gesture-controlled-industrial-automation
pip install -r requirements.txt
```

Update `RAW_PATH` in the notebook to point to your local HGAG dataset directory, then run:

```bash
jupyter notebook hgag_cnn_lstm_final.ipynb
```

---

## 📊 Model Architecture

```
Input: (200, 6)  ← 200 time steps × 6 IMU channels (acc_x/y/z, gyro_x/y/z)
  │
  ├── Conv1D(64, kernel=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv1D(128, kernel=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv1D(128, kernel=3) → BatchNorm → ReLU
  │
  ├── Dropout(0.3)
  ├── LSTM(64)
  ├── Dropout(0.4)
  │
  ├── Dense(64, ReLU)
  ├── Dropout(0.3)
  └── Dense(11, Softmax)  ← 11 gesture classes
```

**Training:** Adam optimizer, EarlyStopping (patience=7), ReduceLROnPlateau, 25 max epochs, batch=128.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design rationale.

---

## 🤖 Gesture → Industrial Command Mapping

| Gesture | Assigned Command | Criticality |
|---|---|---|
| Thumb Up | Start Machine | 8 |
| Fist Making | Emergency Stop | 10 |
| Wrist Extension | Move Arm Right | 6 |
| Wrist Flexion | Move Arm Left | 6 |
| Horizontal Wrist Ext | Activate Tool | 7 |
| Index Finger Flick | Deactivate Tool | 7 |
| Clapping | Speed Up | 5 |
| Coin Flipping | Slow Down | 5 |
| Index Thumb Tap | Confirm Action | 4 |
| Finger Snapping | Cancel Action | 4 |
| Shooting | Call Supervisor | 3 |

Assignment is computed via the **Hungarian algorithm** on a risk cost matrix (gesture error rate × command criticality), minimizing the probability of a misclassified gesture triggering a high-criticality command.

---

## 🔬 Pipeline Modules

The notebook is structured into 9 modules:

| # | Module | Description |
|---|---|---|
| 1 | Problem Understanding / Dataset Analysis | EDA, signal profiles, class balance, audit |
| 2 | Audit-backed Preprocessing | Runtime preprocessing audit across 4 filter modes |
| 3 | Benchmark Baseline | Handcrafted features + Logistic Regression (LOSO) |
| 4 | CNN-LSTM LOSO Training | Full fold-by-fold training with checkpointing |
| 5 | Evaluation / Visualization | Confusion matrix, per-subject accuracy, ROC-AUC |
| 6 | XAI (1D Grad-CAM) | Temporal saliency on representative fold |
| 7 | Deployment / Industrial Readiness | Confidence gating, latency, safe assignment engine |
| 8 | Model Optimization | TFLite dynamic + INT8 quantization |
| 9 | Creative Dashboard + Documentation Exports | Executive KPI dashboard, rubric coverage JSON |

---

## 📈 Key Results

| Metric | Value |
|---|---|
| LOSO Accuracy | See `RESULTS.md` |
| Weighted F1 | See `RESULTS.md` |
| Mean ROC-AUC | See `RESULTS.md` |
| Inference Latency | < 100 ms (per-fold median) |
| Confidence Gate | 0.70 |
| Risk Reduction | vs. naive assignment (Hungarian) |

> Full metrics are exported to `hgag_results_final/final_metrics_summary.json` during the notebook run.

---

## 📦 Output Artifacts

After a full run, `hgag_results_final/` contains:

| File | Description |
|---|---|
| `final_metrics_summary.json` | Consolidated KPI summary |
| `confusion_matrix.png` | LOSO confusion matrix (counts + %) |
| `per_subject_accuracy.png` | Bar chart of per-subject accuracy |
| `roc_auc_per_gesture.png` | Per-gesture ROC-AUC |
| `training_curves_all_folds.png` | Loss/accuracy curves for all folds |
| `xai_gradcam_examples.png` | Grad-CAM saliency on example windows |
| `xai_gradcam_crossfold.png` | Averaged cross-fold Grad-CAM maps |
| `safe_assignment.png` | Risk cost matrix + assignment comparison |
| `inference_latency.png` | Per-fold latency vs 100 ms budget |
| `confidence_entropy.png` | Confidence/entropy distributions |
| `dashboard.png` | 6-panel executive KPI dashboard |
| `quantization_summary.json` | TFLite size/accuracy comparison |
| `deployment_api_contract.json` | REST API schema for live deployment |
| `FINAL_PROJECT_SUMMARY.md` | Auto-generated project report |
| `rubric_coverage_summary.json` | Academic rubric coverage checklist |

---

## 🏭 Deployment Notes

- **Live deployment** requires an upstream IMU driver that segments incoming sensor frames into 200-sample windows at 200 Hz and forwards them to the inference endpoint.
- The confidence gate (default 0.70) rejects ambiguous predictions as `MANUAL_REVIEW`.
- TFLite INT8 model is suitable for edge deployment on microcontrollers or Raspberry Pi.
- See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full API contract and streaming demo design.

---

## 🧪 Configurable Flags

At the top of the notebook, toggle modules:

```python
RUN_BASELINE_BENCHMARK  = True   # Logistic Regression baseline
RUN_XAI                 = True   # 1D Grad-CAM
RUN_MODEL_OPTIMIZATION  = True   # TFLite quantization
RUN_STREAMING_DEMO      = True   # Offline deployment simulation
```

---

## 📚 References

- HGAG Dataset: [Kaggle — hlfbldprnc/hgag-data](https://www.kaggle.com/datasets/hlfbldprnc/hgag-data)
- Grad-CAM: Selvaraju et al., 2017 — *Grad-CAM: Visual Explanations from Deep Networks*
- Hungarian Algorithm: Kuhn, 1955 — *The Hungarian Method for the Assignment Problem*
- TFLite Quantization: [TensorFlow Lite Documentation](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

## 👤 Contributors

- Niloy Sharma (102403422) — Thapar Institute of Engineering & Technology

## 📄 License

This project is created for academic purposes as part of the AI/ML course (UCS321) at Thapar Institute of Engineering & Technology.
