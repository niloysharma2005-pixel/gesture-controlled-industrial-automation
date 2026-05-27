# Results — Evaluation Summary

Full metrics are generated during the notebook run and exported to:
- `hgag_results_final/final_metrics_summary.json`
- `hgag_results_final/FINAL_PROJECT_SUMMARY.md`

This document explains the evaluation methodology and the metrics tracked.

---

## Evaluation Protocol

**Leave-One-Subject-Out (LOSO) Cross-Validation**

- 43 folds total — one per subject
- Each fold: 1 test subject, 4 validation subjects (drawn deterministically), remaining ~38 subjects for training
- **Zero subject leakage**: test subject's data is never used for normalization, training, or validation
- Predictions are accumulated across all folds for global metrics

This is the gold-standard protocol for subject-independent gesture recognition.

---

## Metrics Tracked

### Global

| Metric | Description |
|---|---|
| LOSO Accuracy | Fraction of correctly classified windows across all folds |
| Weighted F1 | F1 score weighted by class support — accounts for imbalance |
| Mean ROC-AUC | Average one-vs-rest AUC across all 11 gesture classes |

### Per-Subject

- Bar chart of per-subject accuracy vs mean accuracy
- Subjects above/below mean highlighted in green/red

### Per-Gesture

- Per-class F1, precision, recall (full classification report)
- Per-class ROC-AUC (one-vs-rest)
- Top 5 most-confused gesture pairs

### Confusion Matrix

- Count matrix (raw misclassification counts)
- Percentage matrix (row-normalized, diagonal = recall per class)

---

## Baseline Comparison

A logistic regression baseline is trained on handcrafted features (time-domain statistics per channel: mean, std, min, max, RMS, zero-crossing rate, percentiles) using the same LOSO protocol.

The CNN-LSTM is compared against this baseline in `benchmark_comparison.png` and `benchmark_comparison.csv`.

---

## Model Calibration

| Metric | Description |
|---|---|
| ECE (Expected Calibration Error) | Weighted average calibration error across confidence bins |
| MCE (Maximum Calibration Error) | Worst-case calibration error across bins |

A reliability diagram is generated (`calibration_reliability.png`) showing confidence vs accuracy per bin.

**High-confidence / high-entropy risk candidates** are also flagged — windows where the model is simultaneously confident (softmax > 0.85) but has high prediction entropy, which may indicate adversarial or ambiguous inputs.

---

## Confidence Gating

The recommended deployment gate is **0.70**.

| Gate | Accepted % | Gated Accuracy |
|---|---|---|
| 0.50 | ~100% | baseline |
| 0.60 | evaluated | evaluated |
| **0.70** | evaluated | **recommended** |
| 0.80 | evaluated | evaluated |
| 0.90 | evaluated | evaluated |

Full table exported to `gating_table.csv`.

---

## XAI — 1D Grad-CAM

Grad-CAM is applied to the last Conv1D layer for each correctly-classified test window.

**Per-fold:** For each gesture class, the highest-confidence correct prediction is selected and its Grad-CAM saliency map is computed and visualised (`xai_gradcam_examples.png`).

**Cross-fold:** Grad-CAM maps are accumulated across all folds and averaged per gesture class (`xai_gradcam_crossfold.png`), providing a dataset-level view of which temporal regions are most informative per gesture.

### What Grad-CAM shows

- High-saliency regions (bright) → time steps the model relies on for that gesture classification
- Expected patterns: acceleration peaks correspond to wrist motion onset/offset; gyroscope patterns vary by gesture type
- Gestures with sharp transients (Finger Snap, Index Flick) are expected to have narrow, localised high-saliency regions
- Gestures with sustained motion (Wrist Extension/Flexion) are expected to have broader saliency distributions

---

## Safe Gesture-Command Assignment

The Hungarian algorithm is applied to a risk cost matrix:

```
cost[i, j] = criticality[command_i] × error_rate[gesture_j]
```

This minimises the expected risk of a misclassified gesture triggering a high-criticality command.

Three cost scenarios are compared:
- **Worst**: worst-case assignment (maximised risk)
- **Naive**: sequential identity assignment
- **Optimal**: Hungarian algorithm minimised assignment

Risk reduction % = `(1 - optimal_cost / naive_cost) × 100`

Exported to `safe_assignment.png` and `safe_assignment_table.csv`.
