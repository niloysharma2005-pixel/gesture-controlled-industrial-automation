# Dataset — HGAG (Hand Gesture Accelerometer and Gyroscope Dataset)

## Source

- **Kaggle:** [hlfbldprnc/hgag-data](https://www.kaggle.com/datasets/hlfbldprnc/hgag-data)
- **Subset used:** HGAG-DATA1

---

## Structure

```
HGAG-DATA/HGAG-DATA1/
├── Clapping/
│   ├── Subject_1/
│   │   └── .csv/
│   │       ├── accel_x_data.csv
│   │       ├── accel_y_data.csv
│   │       ├── accel_z_data.csv
│   │       ├── gyro_x_data.csv
│   │       ├── gyro_y_data.csv
│   │       └── gyro_z_data.csv
│   ├── Subject_2/
│   │   └── ...
│   └── ...
├── Coin Flipping/
├── Finger Snapping/
├── Fist Making/
├── Horizontal Wrist Extension/
├── Index Finger Flick/
├── Index Thumb Tap/
├── Shooting/
├── Thumb Up/
├── Wrist Extension/
└── Wrist Flexion/
```

Each `.csv` file stores **multiple repetitions** as rows (one row = one IMU time-series for one repetition).

---

## Dataset Statistics

| Property | Value |
|---|---|
| Subjects | 43 |
| Gesture classes | 11 |
| IMU channels | 6 (acc_x/y/z, gyro_x/y/z) |
| Sampling rate | 200 Hz |
| Typical trial duration | ~1 second per repetition |

---

## 11 Gesture Classes

| Index | Folder Name | Short Name |
|---|---|---|
| 0 | Clapping | Clapping |
| 1 | Coin Flipping | Coin Flip |
| 2 | Finger Snapping | Finger Snap |
| 3 | Fist Making | Fist Making |
| 4 | Horizontal Wrist Extension | Horiz. Wrist Ext |
| 5 | Index Finger Flick | Idx Finger Flick |
| 6 | Index Thumb Tap | Idx Thumb Tap |
| 7 | Shooting | Shooting |
| 8 | Thumb Up | Thumb Up |
| 9 | Wrist Extension | Wrist Ext |
| 10 | Wrist Flexion | Wrist Flex |

---

## Preprocessing

### Selected Mode: `released`

Four preprocessing modes were evaluated during the runtime preprocessing audit:

| Mode | Description | Outcome |
|---|---|---|
| `released` | Use sensor data as-is (no filter applied) | ✅ Selected |
| `lp90` | 4th-order Butterworth low-pass @ 90 Hz | Candidate |
| `bp30_90` | 4th-order Butterworth band-pass @ 30–90 Hz | Candidate |
| `lp40_accel_detrend` | 40 Hz low-pass + gravity removal from accel | ❌ Rejected |

**Why `released`?** The runtime audit computed RMS retention percentages across all trials for each mode. The `lp40_accel_detrend` mode showed severe signal suppression compared to the released data and was rejected. The `released` mode was chosen as it preserves the full signal as released by the dataset authors, avoiding any filter-introduced distortion.

### Windowing

```
Lead-in offset:  50 samples  (trims pre-gesture quiet period)
Window size:    200 samples  (= 1 second at 200 Hz)
Step size:      200 samples  (non-overlapping — leakage safe)
```

The 50-sample offset was validated by the runtime audit: the quiet lead-in region has significantly lower signal amplitude than the gesture body across all channels, confirming that trimming it improves signal-to-noise without discarding gesture content.

### Per-Fold Normalization

Z-score normalization (mean=0, std=1) is computed **on the inner training set only** per fold and applied to all splits. This prevents any test or validation subject's statistics from influencing normalization.

---

## Preprocessing Audit

The notebook runs a `compute_runtime_audit_summary()` function that:

1. Iterates over all raw trials from disk
2. Computes quiet lead-in ratio per channel (pre-offset vs post-offset amplitude)
3. Computes RMS retention % for all 4 filter modes relative to the released signal
4. Saves results to `runtime_audit_summary.json`

This provides data-driven evidence for every preprocessing decision.

---

## Class Balance

Class balance is evaluated via:
- Coefficient of Variation (CV) across gesture classes
- Gini index

A low CV and Gini confirm that the dataset is approximately balanced across gesture classes, justifying the use of accuracy as a primary metric (supplemented by weighted F1 and ROC-AUC).
