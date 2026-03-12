# lidar_sidewalks

A research pipeline for classifying **Terrestrial Laser Scanning (TLS) point clouds** of European city streets into three surface categories:

| Class code | Label | Description |
|---|---|---|
| `0` | **other** | Vegetation, buildings, vehicles, street furniture, etc. |
| `2` | **sidewalk** | Pavements and pedestrian surfaces |
| `11` | **street** | Carriageway / road surface |

The project progresses from simple exploratory analysis through rule-based geometry, Random Forest classification, feature engineering, and ultimately a segment-first classify-second pipeline validated across multiple cities.

---

## Notebook Guide

Each notebook builds on the previous one. The progression moves from beginner-friendly data exploration through increasingly complex multi-city machine learning.

### 0 — Data Exploration (`0 Data exploration.ipynb`)

**Beginner-friendly starting point.** Opens and inspects a Bologna `.laz` file for the first time — what columns exist, how many points there are, what the bounding box is. Produces 2-D top-down scatter plots (coloured by height and by IFP label) and an interactive 3-D Open3D view. Introduces RANSAC plane detection to find the dominant flat surface (the road). No modelling — this is pure data familiarisation.

> **Key output:** Top-down visualisations of height and classification labels. Understanding of the IFP labelling scheme.

---

### 1 — Carriageway Classification (`1 Carriageway classification.ipynb`)

**First ML model.** Filters to road (class 11) and pavement (class 2) points only; downsamples to a balanced 100 k per class; trains a binary Random Forest on colour + intensity + height. Adds surface normals as an extra geometric feature (normal_z ≈ 1 for horizontal surfaces). Extends to a three-class model (sidewalk / road / other) and optimises hyper-parameters with GridSearchCV. Includes unsupervised clustering (K-Means, DBSCAN) for exploratory analysis.

> **Key output:** Binary and three-class Random Forest classifiers; cross-validated F1 scores; 3-D visualisation of misclassifications.

---

### 2 — Model Class Experiments (`2_model_class_experiments.ipynb`)

**Identifying spatial overfitting.** Runs three structured experiments on the Riga dataset: (A) features without X/Y coordinates — reveals the model's true reliance on spatial position; (B) three-class model with coordinates included; (C) spatial tile-based split — divides the map into a 4×4 grid and holds out one complete tile for testing. Compares random-split vs spatial-split F1 scores side by side. Adds SHAP analysis to explain which features drive each prediction.

> **Key output:** Exposure of spatial overfitting; side-by-side random vs spatial F1 comparison; SHAP feature importance charts.

---

### 3 — Additional Features (`3_additional_features.ipynb`)

**Richer geometry features.** Loads a version of the Bologna point cloud with **37 pre-computed multi-scale geometric descriptors** (eigenvalue-derived: planarity, sphericity, anisotropy, curvature, eigenentropy, roughness at 0.1 m and 0.5 m scales). Re-runs the Experiments A/B/C framework from Notebook 2 on this richer feature set. Uses `psutil` for memory monitoring, feature correlation heatmaps to identify redundant columns, and `fasttreeshap` / SHAP for feature importance analysis.

> **Key output:** Richer feature set; SHAP summary plots evaluating multi-scale geometry vs simple colour/height features.

---

### 4 — Carriageway Focus (`4_carriageway_focus.ipynb`)

**Multi-city generalisation.** Introduces five European cities: Bologna, Riga, Utrecht, Vilnius, and Warsaw. Part 1 builds a tiled single-city (Bologna) baseline with a 65/17.5/17.5 % train/val/test spatial split. Part 2 runs a full **Leave-One-City-Out (LOCO)** evaluation — trains on four cities, tests on the fifth — using eigenvalue geometry features harmonised across all cities. Three targeted controls handle class imbalance: per-fold class weighting, probability override for minority classes, and fold-wise `StandardScaler`. Includes error signature analysis to identify which features drive misclassifications in the hardest cities.

> **Key output:** LOCO summary table (balanced accuracy + macro F1 per held-out city); per-city confusion matrices; error signature diagnostics.

---

### 5 — Deep Learning Exploration (`5_DL_exploration.ipynb`)

**Theory + geometry baseline.** Two halves: (1) a detailed literature review of PointNet++, graph-based segmentation, and the stripe-based approach (Hou & Ai, 2020); (2) a from-scratch geometry baseline — compute per-point geometric features (slope, curvature, roughness, normal direction), group points into **voxel superpoints**, enrich each superpoint with context features from its neighbours, and train a Logistic Regression classifier. Introduces graph smoothing (majority-vote over a k-NN segment graph) to reduce isolated misclassifications.

> **Key output:** KDE feature comparison plots (road vs pavement); segment-level Logistic Regression confusion matrix; balanced accuracy before and after graph smoothing.

---

### 6 — Segment-first, Classify-second (`6_segment_first_classify_second.ipynb`)

**Full multi-city pipeline.** Formalises the segment-then-classify approach from Notebook 5 into a clean, reproducible, multi-city workflow across Riga, Utrecht, Vilnius, and Warsaw. For each city: sample 250 k points → compute local geometric features → build voxel superpoints → enrich with 8-neighbour context features → assign majority-vote ground-truth labels. Then runs LOCO at segment level with a 250-tree Random Forest (`balanced_subsample` class weighting). Provides `segment_first_classify_second()` as a reusable end-to-end inference function, and analyses the top-15 misclassification transitions on the Riga test set.

> **Key output:** Segment-level LOCO summary; end-to-end inference pipeline; Riga confusion matrix + top misclassification transitions.

---

## Source Layout

```
src/
├── data_loader.py              — fetches and pre-processes LAS/LAZ files
├── helpers.py                  — shared utilities (describe_las, etc.)
├── geometry_baseline.py        — per-point feature computation, superpoint segmentation,
│                                 context features, segment majority labelling
├── carriageway_focus_utils.py  — multi-city helpers: eigenvalue extraction, feature
│                                 harmonisation, tiled split, LOCO utilities
└── classification/
    └── rule_based.py           — rule-based geometry classifier (horizontal filter,
                                  region growing, patch classification)

config.py                       — PipelineConfig dataclass with all tunable parameters
```

---

## Quick Start

### 1. Load a point cloud

```python
import laspy
import numpy as np

las = laspy.read("/path/to/cloud.laz")
print(f"Points: {las.header.point_count:,}")
print("Columns:", list(las.point_format.dimension_names))
```

### 2. Run the segment-first classify-second pipeline (Notebook 6 approach)

```python
import numpy as np
from pathlib import Path
import laspy
from sklearn.ensemble import RandomForestClassifier

# Import helpers from src/
from src.geometry_baseline import (
    compute_local_geometric_features,
    build_superpoints_voxel,
    build_segment_context_features,
    assign_segment_majority_labels,
)

# Load and sample
las     = laspy.read("/path/to/city.laz")
xyz     = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
labels  = np.asarray(las.classification)

# 1 — geometric features (k=20 neighbours, 250k sample)
sample_idx, feats = compute_local_geometric_features(xyz, k_neighbors=20, sample_size=250_000)

# 2 — voxel superpoints (0.6 m grid)
sp_id, seg_df = build_superpoints_voxel(xyz[sample_idx], feats, voxel_size=0.6, min_points=8)

# 3 — context features (8 nearest neighbour segments)
seg_ctx = build_segment_context_features(seg_df, n_neighbors=8)

# 4 — ground-truth labels per segment
seg_lbl = assign_segment_majority_labels(sp_id, labels[sample_idx])

# 5 — prepare training set
seg = seg_ctx.merge(seg_lbl, on="segment_id", how="inner")
feature_cols = [c for c in seg.columns if c not in {"segment_id","target_label","target_purity"}]
X, y = seg[feature_cols].values, seg["target_label"].values

# 6 — train classifier (three classes: 0=other, 2=sidewalk, 11=street)
clf = RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", n_jobs=-1)
clf.fit(X, y)
```

### 3. Save predictions

```python
y_pred = clf.predict(X)

# Map predictions back to points (each point → its segment's prediction)
seg_pred_map = dict(zip(seg["segment_id"], y_pred))
point_labels = np.array([seg_pred_map.get(sid, 0) for sid in sp_id])

# Write back to LAS
las.add_extra_dim(laspy.ExtraBytesParams(name="pred_label", type=np.uint8))
las.pred_label = point_labels
las.write("/path/to/cloud_with_labels.laz")
```

---

## Recommended Notebook Order

If you are new to the project, work through the notebooks in sequence:

```
0 Data exploration            ← start here; no modelling, just look at the data
1 Carriageway classification  ← first ML model (Random Forest)
2 Model class experiments     ← spatial overfitting diagnosis
3 Additional features         ← richer multi-scale geometry features + SHAP
4 Carriageway focus           ← multi-city LOCO with eigenvalue features
5 DL exploration              ← theory + geometry baseline prototype
6 Segment-first classify-second ← production-ready multi-city pipeline
```

---

## Dependencies

Key Python packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `laspy` | Read/write `.las` / `.laz` point cloud files |
| `numpy` | Numerical array operations |
| `pandas` | Tabular data |
| `scikit-learn` | Machine learning (Random Forest, LogReg, StandardScaler, metrics) |
| `open3d` | 3-D point cloud visualisation and geometry |
| `matplotlib` / `seaborn` | Plotting |
| `shap` / `fasttreeshap` | Model explainability |
| `psutil` | Memory monitoring |
| `dotenv` / `upath` | Environment config and remote file access |

---

## Notes on Performance

- Most notebooks sample **250 000 – 1 000 000 points** to remain memory-safe. For full-cloud runs, increase `sample_size` incrementally.
- Set `n_jobs=-1` on machines with many CPU cores (normal estimation and Random Forest training both parallelise well).
- The segment-level pipeline (Notebook 6) is **significantly faster** than per-point classification because it reduces hundreds of millions of points to thousands of segments before classifying.
- Expect training to take **2–15 minutes** per LOCO fold depending on sample size and number of trees. Use `n_estimators=100` for quick experiments and `n_estimators=250` for final runs.


