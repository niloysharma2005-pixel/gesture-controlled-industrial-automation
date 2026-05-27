# Architecture — CNN-LSTM for IMU Gesture Recognition

This document explains every architectural decision in the `hgag_cnn_lstm_final.ipynb` model.

---

## Overview

The model combines **1D Convolutional layers** (local temporal feature extraction) and an **LSTM layer** (sequential dependency modelling) to classify 11 hand gestures from 200-sample × 6-channel IMU windows.

---

## Input Shape

```
(batch, 200, 6)
```

- **200 time steps** → 1 second at 200 Hz (with 50-sample offset trim)
- **6 channels** → acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

---

## Layer-by-Layer Rationale

### Conv1D Block × 3 (kernels: 5, 5, 3 — filters: 64, 128, 128)

Extracts **local temporal motions** before recurrent modelling.

- Kernel size **5** ≈ 25 ms at 200 Hz — appropriate for short gesture transients and wrist-motion bursts.
- Kernel size **3** in the third layer refines finer temporal details after broader local patterns have already been captured by the first two layers.
- Filters double from 64 → 128 to progressively increase representational capacity.

### MaxPooling1D × 2 (pool size: 2)

Compresses the temporal axis: **200 → 100 → 50 steps** before the LSTM.

- Reduces recurrent cost and noise sensitivity.
- Preserves enough motion structure for sequence modelling.
- The 50-step sequence passed to the LSTM represents ~250 ms of gesture signal at 200 Hz.

### BatchNormalization

Stabilises training across subjects whose motion amplitudes differ even after per-fold train-only normalization.

Particularly important in a **LOSO setting** where each fold has a different test subject and potentially different signal magnitude ranges.

### LSTM (64 hidden units)

Models **inter-frame dependencies** that local convolutions cannot capture alone (e.g., the temporal ordering of acceleration peaks and gyro spikes within a gesture).

- 64 hidden units provide enough sequence capacity without being large enough to memorise subject-specific patterns in a 43-subject dataset.
- Overfitting risk is real in LOSO settings with only ~38–39 training subjects per fold.

### Dropout (0.3 / 0.4 / 0.3)

Conservative regularisation with slightly stronger dropout around the recurrent and dense stages to reduce subject memorisation.

| Location | Rate |
|---|---|
| After Conv blocks | 0.3 |
| After LSTM | 0.4 |
| After Dense(64) | 0.3 |

### Dense(64, ReLU) → Dense(11, Softmax)

- 64-unit dense layer provides non-linear mixing of the LSTM's sequence summary.
- 11-class Softmax output → probability distribution over gestures.

---

## Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam | Robust for medium-scale time-series tasks |
| Max Epochs | 25 | Sufficient given EarlyStopping |
| Batch Size | 128 | Balances GPU utilisation and gradient noise |
| EarlyStopping patience | 7 | Allows LR reduction to take effect before stopping |
| ReduceLROnPlateau factor | 0.5, patience=3 | Smooths late convergence |
| Loss | Sparse Categorical Crossentropy | Standard for integer-label multiclass |

---

## Data Augmentation (Train-Only)

All augmentations are applied **after per-fold z-score normalization** and only to training data:

| Augmentation | Parameters |
|---|---|
| Gaussian jitter | σ = 0.03 |
| Amplitude scaling (accel) | ×[0.85, 1.15] |
| Amplitude scaling (gyro) | ×[0.90, 1.10] |
| Local window warp | 25% of window, factor [0.8, 1.2] |

Gyro scaling range is tighter than accelerometer because gyroscope signals are more sensitive to scaling distortion.

---

## Windowing Parameters

| Parameter | Value | Notes |
|---|---|---|
| Window size | 200 samples | = 1 second at 200 Hz |
| Step size | 200 samples | Non-overlapping (leakage-safe) |
| Lead-in offset | 50 samples | Trims quiet pre-gesture period confirmed by runtime audit |

---

## LOSO Protocol

- **Test set:** 1 held-out subject (never seen during training or normalization)
- **Validation set:** 4 subjects drawn from remaining training subjects (deterministic per fold)
- **Inner train set:** Remaining ~38 subjects
- **Normalization:** Computed on inner training set only; applied to val and test sets
- **Number of folds:** 43 (one per subject)

This ensures **zero subject leakage** — the most stringent evaluation protocol for biometric/gesture datasets.

---

## XAI: 1D Grad-CAM

Grad-CAM is applied on the **last Conv1D layer** to produce a temporal saliency map for each gesture class.

For each correctly-classified test window, the gradient of the class score with respect to the Conv1D feature maps is used to weight the feature map activations, producing a 1D heatmap over time.

Cross-fold Grad-CAM averages saliency maps across all folds for each gesture class, giving a dataset-level view of which temporal regions each gesture relies on most.
