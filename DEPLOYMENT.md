# Deployment — Industrial Readiness Guide

## Overview

The pipeline exports a deployment-ready inference system with:

- Confidence-gated softmax classifier
- TFLite dynamic + INT8 quantized models
- REST API contract for live integration
- Offline streaming simulation

---

## TFLite Model Export

Two quantized models are exported from the representative fold:

| Model | Type | Description |
|---|---|---|
| `representative_fold_model_dynamic.tflite` | Dynamic quantization | Weights quantized to INT8 at inference time |
| `representative_fold_model_int8.tflite` | Full INT8 quantization | Both weights and activations quantized |

Quantization summary (size reduction, accuracy delta) is saved to `quantization_summary.json`.

### Suitable For

- Raspberry Pi 4 / Zero 2W
- NVIDIA Jetson Nano
- STM32 with TFLite Micro (INT8 only)
- Any edge device with 200 Hz IMU data acquisition

---

## Inference API Contract

**Endpoint:** `POST /predict`

### Request Schema

```json
{
  "window": {
    "type": "array",
    "shape": [200, 6],
    "dtype": "float32",
    "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    "units": {
      "accel": "m/s^2 (device-released or deployment-side normalized)",
      "gyro": "rad/s (device-released or deployment-side normalized)"
    }
  }
}
```

### Response Schema

```json
{
  "gesture": "string",
  "confidence": "float [0, 1]",
  "accepted": "bool (confidence >= gate)",
  "assigned_command": "string | MANUAL_REVIEW",
  "entropy_nats": "float"
}
```

### Parameters

| Parameter | Value |
|---|---|
| Confidence gate | 0.70 |
| Latency budget | 100 ms |
| Window size | 200 samples @ 200 Hz |
| Channels | 6 (3-axis accel + 3-axis gyro) |

Full contract exported to `deployment_api_contract.json`.

---

## Confidence Gating

Predictions with `max(softmax) < 0.70` are flagged as `MANUAL_REVIEW` and **not** forwarded to the command actuator.

This is critical in industrial settings: it is safer to reject an ambiguous gesture and require the operator to repeat it than to execute an incorrect high-criticality command.

```python
if max_confidence >= CONFIDENCE_GATE:
    execute_command(assigned_command)
else:
    request_manual_review()
```

---

## Latency

Inference latency is measured as median of 50 single-window forward passes per fold.

**Budget:** 100 ms (suitable for reactive human-robot interaction)

At 200 Hz, each window = 1 second of sensor data. The inference call takes << 100 ms on modern hardware, leaving ample time for upstream signal acquisition and downstream command routing.

Latency results are saved to `inference_latency.png`.

---

## Live Deployment Requirements

The offline simulation (`deployment_simulation_replay.csv`) replays stored test-set windows. For a **live system**, the upstream pipeline must:

1. **IMU Driver** — Continuously sample acc/gyro at 200 Hz via serial/BLE
2. **Windowing Buffer** — Accumulate 200 samples, apply 50-sample offset trim
3. **Normalization** — Apply the per-fold training-set mean/std (or a deployment-calibrated z-score)
4. **Inference** — Call the TFLite model with the 200×6 float32 window
5. **Confidence Gate** — Route to command actuator or MANUAL_REVIEW
6. **Command Actuator** — Send the assigned industrial command to the robot controller

---

## Streaming Demo

The offline streaming simulation (`RUN_STREAMING_DEMO=True`) replays the full test-set window sequence through the gating pipeline and exports:

- `deployment_simulation_replay.csv` — 500 simulated prediction frames
- `deployment_api_example.json` — Example input/output pair

---

## Safety Notes

- The safe gesture-command assignment (Hungarian algorithm) should be **recomputed** whenever the model is retrained, as error rates per gesture may change.
- Any gesture assigned to `EMERGENCY STOP` (criticality=10) should be the gesture with the **lowest error rate** in the deployed fold ensemble.
- For certified industrial deployment, the system should undergo IEC 62061 / ISO 13849 safety assessment.
