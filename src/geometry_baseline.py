from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_local_geometric_features(
    points_xyz: np.ndarray,
    k_neighbors: int = 20,
    sample_size: int | None = 300_000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    if points_xyz.shape[0] < 4:
        raise ValueError("at least 4 points are required")

    n = points_xyz.shape[0]
    if sample_size is None or sample_size >= n:
        sample_idx = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))

    sample_pts = points_xyz[sample_idx]
    k = min(max(k_neighbors, 4), sample_pts.shape[0])

    knn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=n_jobs)
    knn.fit(sample_pts)
    neigh_idx = knn.kneighbors(sample_pts, return_distance=False)

    local = sample_pts[neigh_idx]
    centered = local - local.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0

    eigsum = np.maximum(eigvals.sum(axis=1), 1e-12)
    curvature = eigvals[:, 0] / eigsum
    roughness = np.sqrt(np.maximum(eigvals[:, 0], 0.0))
    nz = np.abs(normals[:, 2])
    slope_deg = np.degrees(np.arccos(np.clip(nz, 0.0, 1.0)))
    local_height_std = local[:, :, 2].std(axis=1)

    features = {
        "slope_deg": slope_deg,
        "curvature": curvature,
        "roughness": roughness,
        "local_height_std": local_height_std,
        "abs_nz": nz,
    }
    return sample_idx, features


def summarize_features_by_class(
    features: dict[str, np.ndarray],
    sampled_labels: np.ndarray,
    class_map: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for class_id, class_name in class_map.items():
        mask = sampled_labels == class_id
        support = int(mask.sum())
        if support == 0:
            continue
        for feat_name, feat_values in features.items():
            vals = feat_values[mask]
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": feat_name,
                    "support": support,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                }
            )
    return pd.DataFrame(rows)


def build_superpoints_voxel(
    points_xyz: np.ndarray,
    point_features: dict[str, np.ndarray],
    voxel_size: float = 0.25,
    min_points: int = 30,
) -> tuple[np.ndarray, pd.DataFrame]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    n = points_xyz.shape[0]
    if n == 0:
        raise ValueError("points_xyz must not be empty")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    for name, values in point_features.items():
        if values.shape[0] != n:
            raise ValueError(f"feature '{name}' must have length N")

    v = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, inverse = np.unique(v, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    keep_mask = counts[inverse] >= min_points
    superpoint_id = np.full(n, -1, dtype=np.int64)

    kept_old_ids = np.where(counts >= min_points)[0]
    remap = np.full(counts.shape[0], -1, dtype=np.int64)
    remap[kept_old_ids] = np.arange(kept_old_ids.shape[0], dtype=np.int64)
    superpoint_id[keep_mask] = remap[inverse[keep_mask]]

    rows: list[dict] = []
    for old_id in kept_old_ids:
        new_id = int(remap[old_id])
        mask = inverse == old_id
        pts = points_xyz[mask]
        row = {
            "segment_id": new_id,
            "n_points": int(mask.sum()),
            "cx": float(np.mean(pts[:, 0])),
            "cy": float(np.mean(pts[:, 1])),
            "cz": float(np.mean(pts[:, 2])),
            "z_range": float(np.max(pts[:, 2]) - np.min(pts[:, 2])),
        }
        for feat_name, feat_values in point_features.items():
            vals = feat_values[mask]
            row[f"{feat_name}_mean"] = float(np.mean(vals))
            row[f"{feat_name}_median"] = float(np.median(vals))
            row[f"{feat_name}_p90"] = float(np.percentile(vals, 90))
        rows.append(row)

    segment_df = pd.DataFrame(rows)
    return superpoint_id, segment_df


def build_segment_context_features(
    segment_df: pd.DataFrame,
    n_neighbors: int = 8,
) -> pd.DataFrame:
    if segment_df.empty:
        return segment_df.copy()

    required = ["segment_id", "cx", "cy", "cz"]
    for col in required:
        if col not in segment_df.columns:
            raise ValueError(f"segment_df missing required column: {col}")

    out = segment_df.copy().reset_index(drop=True)
    coords = out[["cx", "cy", "cz"]].to_numpy(dtype=float)
    k = min(max(2, n_neighbors + 1), coords.shape[0])

    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    knn.fit(coords)
    dist, neigh = knn.kneighbors(coords, return_distance=True)

    feature_cols = [
        c
        for c in out.columns
        if c not in {"segment_id", "n_points", "cx", "cy", "cz"}
    ]

    for col in feature_cols:
        vals = out[col].to_numpy(dtype=float)
        neigh_vals = vals[neigh[:, 1:]]
        out[f"ctx_{col}_mean"] = neigh_vals.mean(axis=1)
        out[f"ctx_{col}_std"] = neigh_vals.std(axis=1)
        out[f"delta_{col}"] = vals - neigh_vals.mean(axis=1)

    out["ctx_mean_dist"] = dist[:, 1:].mean(axis=1)
    out["ctx_min_dist"] = dist[:, 1:].min(axis=1)
    return out


def assign_segment_majority_labels(
    superpoint_id: np.ndarray,
    point_labels: np.ndarray,
) -> pd.DataFrame:
    if superpoint_id.shape[0] != point_labels.shape[0]:
        raise ValueError("superpoint_id and point_labels must have same length")

    valid = superpoint_id >= 0
    seg_ids = superpoint_id[valid]
    labels = point_labels[valid]

    rows: list[dict] = []
    for seg_id in np.unique(seg_ids):
        mask = seg_ids == seg_id
        lab = labels[mask]
        uniq, cnt = np.unique(lab, return_counts=True)
        best = int(uniq[np.argmax(cnt)])
        purity = float(np.max(cnt) / np.sum(cnt))
        rows.append(
            {
                "segment_id": int(seg_id),
                "target_label": best,
                "target_purity": purity,
            }
        )
    return pd.DataFrame(rows)


def prepare_segment_ml_matrix(
    segment_ctx_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    keep_labels: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    merged = segment_ctx_df.merge(labels_df, on="segment_id", how="inner")
    merged = merged[merged["target_label"].isin(keep_labels)].copy()
    if merged.empty:
        raise ValueError("No segments found for requested keep_labels")

    y = merged["target_label"].to_numpy(dtype=int)
    drop_cols = {"segment_id", "target_label"}
    feature_cols = [c for c in merged.columns if c not in drop_cols]
    x = merged[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, feature_cols, scaler
