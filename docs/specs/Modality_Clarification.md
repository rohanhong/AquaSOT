# Modality Concept Clarification

## CRITICAL: Two Different Meanings of "Modality"

The term "modality" in AquaSOT-3D has been **incorrectly used** to mean two different things. This document clarifies the distinction.

---

## ❌ INCORRECT Usage: Sensor Selection Modality

**What it meant (WRONG)**:
- Choosing which sensors to use (RGB, Sonar, or both)
- Implemented via `modalities=['rgb', 'sonar']` parameter in datasets
- Allowed "RGB-only" or "Sonar-only" training

**Why this is wrong**:
- AquaSOT-3D is designed as a **multi-modal fusion tracker**
- The entire architecture (G²-DCMA fusion, dual-view heads) requires **both RGB and Sonar inputs**
- Single-sensor operation defeats the purpose of the system

**Status**: **REMOVED** - All `modalities` parameters deleted from codebase

---

## ✅ CORRECT Usage: Target Visibility Modality

**What it means (CORRECT)**:
- Whether the **target object** is visible in RGB and/or Sonar images
- Independent of sensor hardware existence
- Handled via validity flags: `is_rgb_valid`, `is_sonar_valid`, `has_object_rgb`, `has_object_sonar`

### Four Target Visibility States:

| State | Description | RGB Valid | Sonar Valid | Use Case |
|-------|-------------|-----------|-------------|----------|
| **Zero-Modality** | Target not visible in either sensor | ❌ False | ❌ False | Complete occlusion |
| **Single-RGB** | Target visible only in RGB | ✅ True | ❌ False | Sonar occlusion/noise |
| **Single-Sonar** | Target visible only in Sonar | ❌ False | ✅ True | RGB occlusion/poor visibility |
| **Dual-Modality** | Target visible in both sensors | ✅ True | ✅ True | Optimal tracking condition |

---

## System Architecture

### Data Flow (Always Dual-Sensor):

```
RGB Image (always present)    Sonar Image (always present)
    ↓                              ↓
RGB Backbone (ViT)            Sonar Encoder (ViT)
    ↓                              ↓
RGB Features                  Sonar Features
    ↓                              ↓
    └──────────┬──────────┘
               ↓
      G²-DCMA Fusion Module
      (conditional routing based on validity)
               ↓
         ┌─────┴─────┐
         ↓           ↓
    RGB BoxHead  Sonar BoxHead
```

### Conditional Fusion Logic:

```python
# In ConditionalG2DCMA.forward()
if is_rgb_valid and is_sonar_valid:
    # Route 1: Dual-modality → Full G²-DCMA fusion
    fused = self.g2_dcma_fusion(rgb_features, sonar_features, metadata)

elif is_rgb_valid and not is_sonar_valid:
    # Route 2: Single-RGB → RGB bypass
    fused = self.rgb_bypass(rgb_features)

elif not is_rgb_valid and is_sonar_valid:
    # Route 3: Single-Sonar → Sonar bypass
    fused = self.sonar_bypass(sonar_features)

else:  # not is_rgb_valid and not is_sonar_valid
    # Route 4: Zero-modality → Predict no target
    # Confidence head should output low confidence
    fused = zeros_like(rgb_features)
```

---

## Code Annotations

### Correct Patterns:

```python
# ✅ CORRECT: Target visibility flags
is_rgb_valid = not is_occluded(rgb_bbox)  # Target visible in RGB?
is_sonar_valid = not is_occluded(sonar_bbox)  # Target visible in Sonar?

# ✅ CORRECT: Validity-based routing
zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)
single_rgb_mask = is_rgb_valid & (~is_sonar_valid)
single_sonar_mask = (~is_rgb_valid) & is_sonar_valid
dual_modality_mask = is_rgb_valid & is_sonar_valid

# ✅ CORRECT: Always load both sensors
rgb_img = load_image(rgb_path)  # Always executed
sonar_img = load_image(sonar_path)  # Always executed
```

### Incorrect Patterns (TO BE REMOVED):

```python
# ❌ WRONG: Sensor selection parameter
if 'rgb' in self.modalities:  # BAD - should always process RGB
    rgb_img = load_image(rgb_path)

if 'sonar' in self.modalities:  # BAD - should always process Sonar
    sonar_img = load_image(sonar_path)

# ❌ WRONG: Optional sensor configs
modalities = ['rgb']  # BAD - missing Sonar
modalities = ['sonar']  # BAD - missing RGB
```

---

## Dataset Annotation Semantics

### Groundtruth Format:

```
# groundtruth.txt
x,y,w,h   # Regular bounding box
0,0,0,0   # Special case: target is occluded/not visible
```

### Interpretation:

- **Regular bbox (x,y,w,h)**: Target is **visible** in this modality
  - Sets `is_<modality>_valid = True`
  - Sets `has_object_<modality> = True`

- **Occlusion bbox (0,0,0,0)**: Target is **not visible** in this modality
  - Sets `is_<modality>_valid = False`
  - Sets `has_object_<modality> = False`

### Example Scenarios:

| RGB GT | Sonar GT | Interpretation | Modality State |
|--------|----------|----------------|----------------|
| (100,50,30,40) | (120,60,35,45) | Target visible in both | **Dual-modality** |
| (100,50,30,40) | (0,0,0,0) | Target visible only in RGB | **Single-RGB** |
| (0,0,0,0) | (120,60,35,45) | Target visible only in Sonar | **Single-Sonar** |
| (0,0,0,0) | (0,0,0,0) | Target not visible | **Zero-modality** |

---

## Training Implications

### Loss Computation:

```python
# RGB box loss (only when RGB has valid target)
if has_object_rgb:
    loss_rgb_box = compute_bbox_loss(pred_rgb_box, gt_rgb_box)
else:
    loss_rgb_box = 0.0  # Skip loss for occluded modality

# Sonar box loss (only when Sonar has valid target)
if has_object_sonar:
    loss_sonar_box = compute_bbox_loss(pred_sonar_box, gt_sonar_box)
else:
    loss_sonar_box = 0.0  # Skip loss for occluded modality

# Consistency loss (only for dual-modality samples)
if has_object_rgb and has_object_sonar:
    loss_consistency = compute_consistency_loss(rgb_features, sonar_features)
else:
    loss_consistency = 0.0  # No consistency when single-modality
```

### Confidence Head:

The confidence head learns to predict **target presence probability**:
- High confidence → Target is present and visible
- Low confidence → Target is occluded or absent

This allows the tracker to **reject false positives** in zero-modality cases.

---

## Summary

### What Changed:

| Concept | Old (Wrong) | New (Correct) |
|---------|-------------|---------------|
| **Modality** | Sensor type selection | Target visibility state |
| **Dataset `modalities`** | `['rgb', 'sonar']` parameter | **REMOVED** - always dual-sensor |
| **Backbone config** | Optional RGB/Sonar | **REQUIRED** - both mandatory |
| **Fusion logic** | Skipped if sensor disabled | **Conditional** based on target validity |

### Key Principle:

> **Sensors are always present. Target visibility varies.**

---

## References

- Tracker implementation: `src/aquasot3d/models/tracker.py::detect_modality_availability()`
- Fusion routing: `src/aquasot3d/models/fusion/g2_dcma.py::ConditionalG2DCMA.forward()`
- Dataset validity: `src/aquasot3d/dataloader/datasets/rgb_sonar_dataset.py::_get_frame_data()`
