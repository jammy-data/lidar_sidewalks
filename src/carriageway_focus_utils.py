from __future__ import annotations

"""Reusable helpers for carriageway-focused classification notebooks.

This module centralizes the data schema handling and evaluation helpers used in
`Notebooks/4_carriageway_focus.ipynb`.

Key responsibilities:
- map raw LAS classes into the 3-class target space,
- harmonize city files with heterogeneous feature schemas,
- derive comparable eigenvalue-based geometric descriptors,
- align feature spaces across cities for fair LOCO evaluation,
- provide common plotting and ablation helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


REQUIRED_FIELDS = [
    "X",
    "Y",
    "Z",
    "intensity",
    "classification",
    "red",
    "green",
    "blue",
    "height_division",
    "Roughness (0.1)",
    "Normal change rate (0.1)",
    "Surface density (r=0.1)",
    "Volume density (r=0.1)",
    "Planarity (0.1)",
    "Sphericity (0.1)",
    "Verticality (0.1)",
    "Roughness (0.05)",
    "Normal change rate (0.05)",
    "Surface density (r=0.05)",
    "Volume density (r=0.05)",
    "Planarity (0.05)",
    "Sphericity (0.05)",
    "Verticality (0.05)",
    "Roughness (0.5)",
    "Normal change rate (0.5)",
    "Surface density (r=0.5)",
    "Volume density (r=0.5)",
    "Planarity (0.5)",
    "Sphericity (0.5)",
    "Verticality (0.5)",
    "Roughness (1)",
    "Normal change rate (1)",
    "Surface density (r=1)",
    "Volume density (r=1)",
    "Planarity (1)",
    "Sphericity (1)",
    "Verticality (1)",
]

LABEL_MAP_THREE = {0: "other", 2: "sidewalk", 11: "street"}

BASE_COMPAT_FIELDS = [
    "X",
    "Y",
    "Z",
    "intensity",
    "return_number",
    "number_of_returns",
    "scan_angle_rank",
    "red",
    "green",
    "blue",
    "height_division",
    "classification",
]

MIN_REQUIRED_FIELDS = ["X", "Y", "Z", "classification"]

EIGEN_SCALES = [0.1, 0.5, 1.0]


def map_to_three_classes(classification_array: np.ndarray) -> np.ndarray:
    """Map raw LAS classification codes to a 3-class target.

    Mapping used throughout this project:
    - 2  -> sidewalk
    - 11 -> street
    - all others -> other (0)
    """
    return np.where(
        classification_array == 2,
        2,
        np.where(classification_array == 11, 11, 0),
    ).astype(np.uint8)


def las_to_three_class_df(
    las_obj,
    city_name: str,
    required_columns: list[str],
    n_sample: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build a city DataFrame using an explicit required schema.

    This strict loader is useful when all expected engineered dimensions are
    guaranteed to exist.
    """
    available_dims = set(las_obj.point_format.dimension_names)
    missing = [col for col in required_columns if col not in available_dims]
    if missing:
        raise ValueError(f"{city_name} is missing columns: {missing}")

    data = {col: np.asarray(getattr(las_obj, col)) for col in required_columns}
    city_df = pd.DataFrame(data)
    city_df["target"] = map_to_three_classes(city_df["classification"].to_numpy())
    city_df["city"] = city_name

    if n_sample is not None and n_sample < len(city_df):
        city_df = city_df.sample(n=n_sample, random_state=random_seed).copy()

    return city_df


def _scale_suffix(scale: float) -> str:
    """Convert numeric scale to column-safe suffix (e.g. 0.5 -> '0p5')."""
    return str(scale).replace(".", "p")


def _eigen_triplet_exists(available_dims: set[str], scale: float) -> bool:
    """Return True when all three eigenvalue dimensions exist for a scale."""
    return all(
        f"{k} eigenvalue ({scale:g})" in available_dims
        for k in ("1st", "2nd", "3rd")
    )


def extract_eigen_triplets_from_las(
    las_obj,
    scales: list[float] | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract LAS-provided eigenvalue triplets for requested scales.

    Returns
    -------
    dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Mapping scale -> (lambda1, lambda2, lambda3), where lambda1 >= lambda2 >= lambda3.
        Only scales with complete triplets in the LAS are returned.
    """
    scales = scales or EIGEN_SCALES
    dims = set(las_obj.point_format.dimension_names)

    out: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scale in scales:
        if not _eigen_triplet_exists(dims, scale):
            continue
        l1 = np.asarray(getattr(las_obj, f"1st eigenvalue ({scale:g})"), dtype=np.float64)
        l2 = np.asarray(getattr(las_obj, f"2nd eigenvalue ({scale:g})"), dtype=np.float64)
        l3 = np.asarray(getattr(las_obj, f"3rd eigenvalue ({scale:g})"), dtype=np.float64)
        out[scale] = (l1, l2, l3)
    return out


def derive_geometry_features_from_eigen_triplets(
    eigen_by_scale: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Derive harmonized geometric features from eigenvalue triplets.

    Parameters
    ----------
    eigen_by_scale : dict
        Mapping `scale -> (lambda1, lambda2, lambda3)`.

    Returns
    -------
    dict[str, np.ndarray]
        Derived features keyed by harmonized names like
        `planarity_0p5`, `curvature_1p0`, `roughness_0p1`, etc.
    """
    eps = 1e-12
    features: dict[str, np.ndarray] = {}

    for scale, (l1, l2, l3) in eigen_by_scale.items():
        l1 = np.asarray(l1, dtype=np.float64)
        l2 = np.asarray(l2, dtype=np.float64)
        l3 = np.asarray(l3, dtype=np.float64)

        l1_safe = np.maximum(l1, eps)
        lam_sum = np.maximum(l1 + l2 + l3, eps)
        p1 = l1 / lam_sum
        p2 = l2 / lam_sum
        p3 = l3 / lam_sum
        suffix = _scale_suffix(scale)

        features[f"eig1_{suffix}"] = l1.astype(np.float32)
        features[f"eig2_{suffix}"] = l2.astype(np.float32)
        features[f"eig3_{suffix}"] = l3.astype(np.float32)
        features[f"planarity_{suffix}"] = ((l2 - l3) / l1_safe).astype(np.float32)
        features[f"sphericity_{suffix}"] = (l3 / l1_safe).astype(np.float32)
        features[f"anisotropy_{suffix}"] = ((l1 - l3) / l1_safe).astype(np.float32)
        features[f"curvature_{suffix}"] = (l3 / lam_sum).astype(np.float32)
        features[f"eigenentropy_{suffix}"] = (
            -(p1 * np.log(np.maximum(p1, eps)) + p2 * np.log(np.maximum(p2, eps)) + p3 * np.log(np.maximum(p3, eps)))
        ).astype(np.float32)
        features[f"roughness_{suffix}"] = np.sqrt(np.maximum(l3, 0.0)).astype(np.float32)

    return features


def build_harmonized_city_df(
    las_obj,
    city_name: str,
    n_sample: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a harmonized per-city feature table from heterogeneous LAS schemas.

    Strategy:
    1) keep present base fields,
    2) include raw roughness columns when available,
    3) derive consistent eigen descriptors from eigenvalue triplets,
    4) add project target labels.

    This function intentionally tolerates missing optional fields while still
    enforcing minimal geometry/classification availability.
    """
    available_dims = set(las_obj.point_format.dimension_names)

    missing_min = [col for col in MIN_REQUIRED_FIELDS if col not in available_dims]
    if missing_min:
        raise ValueError(f"{city_name} missing minimal required fields: {missing_min}")

    present_base = [col for col in BASE_COMPAT_FIELDS if col in available_dims]
    df = pd.DataFrame({col: np.asarray(getattr(las_obj, col)) for col in present_base})

    # Harmonized vertical feature for comparability across cities.
    # Prefer LAS-provided normalized height when available; otherwise create a
    # city-centered fallback from raw Z.
    if "height_division" in df.columns:
        df["z_norm"] = np.asarray(df["height_division"], dtype=np.float32)
    else:
        z = np.asarray(df["Z"], dtype=np.float64)
        df["z_norm"] = (z - np.median(z)).astype(np.float32)

    # Two-step eigen workflow:
    # 1) extract raw eigenvalue triplets from LAS,
    # 2) derive harmonized geometric features from those triplets.
    eigen_by_scale = extract_eigen_triplets_from_las(las_obj, scales=EIGEN_SCALES)
    derived = derive_geometry_features_from_eigen_triplets(eigen_by_scale)
    for col, values in derived.items():
        df[col] = values

    # Target + city label
    df["classification"] = df["classification"].astype(np.uint8)
    df["target"] = map_to_three_classes(df["classification"].to_numpy())
    df["city"] = city_name

    # Compact dtypes
    for col in ["intensity", "red", "green", "blue"]:
        if col in df.columns:
            df[col] = df[col].astype(np.uint16)

    if n_sample is not None and n_sample < len(df):
        df = df.sample(n=n_sample, random_state=random_seed).copy()
    print(f"{city_name}: {len(df)} points, {len(df.columns)} features (including target)")
    return df
    


def align_city_feature_space(
    city_dfs: dict[str, pd.DataFrame],
    protected_cols: tuple[str, ...] = ("target", "city", "classification"),
    drop_raw_z_if_z_norm: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Align multiple city tables to their strict common feature intersection.

    Ensures every city is trained/evaluated with exactly the same predictors,
    which is required for fair cross-city comparability.
    """
    if not city_dfs:
        raise ValueError("city_dfs must not be empty")

    city_names = list(city_dfs.keys())
    feature_sets = []
    for city_name in city_names:
        cols = [c for c in city_dfs[city_name].columns if c not in protected_cols]
        feature_sets.append(set(cols))

    common_features = sorted(set.intersection(*feature_sets))
    if len(common_features) == 0:
        raise ValueError("No common feature columns across cities after harmonization")

    if drop_raw_z_if_z_norm and "z_norm" in common_features and "Z" in common_features:
        common_features = [f for f in common_features if f != "Z"]

    aligned = {}
    keep_cols = common_features + [c for c in protected_cols if c in city_dfs[city_names[0]].columns]
    for city_name, df in city_dfs.items():
        city_keep = common_features + [c for c in protected_cols if c in df.columns]
        aligned[city_name] = df[city_keep].copy()

    return aligned, common_features


def balance_tile(
    df_tile: pd.DataFrame,
    target_col: str = "target",
    random_state: int = 42,
) -> pd.DataFrame:
    """Downsample each class inside one tile to the smallest class count."""
    counts = df_tile[target_col].value_counts()
    n_min = counts.min()
    return df_tile.groupby(target_col, group_keys=False).sample(
        n=n_min,
        random_state=random_state,
    )


def build_balanced_tile_split(
    df_all: pd.DataFrame,
    tile_ids,
    tile_col: str = "tile_id",
    target_col: str = "target",
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a split DataFrame by balancing class counts independently per tile."""
    chunks = []
    for tile_id in sorted(tile_ids):
        tile_df = df_all[df_all[tile_col] == tile_id]
        if tile_df.empty:
            continue
        chunks.append(balance_tile(tile_df, target_col=target_col, random_state=random_state))
    if not chunks:
        return pd.DataFrame(columns=df_all.columns)
    return pd.concat(chunks, ignore_index=True)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Render a labeled confusion matrix heatmap (raw or row-normalized)."""
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        cbar_label = "Normalized Count"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": cbar_label},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.show()


def run_loco_with_class_weight(
    city_dfs: dict[str, pd.DataFrame],
    model_feature_cols: list[str],
    class_weight,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run LOCO evaluation for one class-weight strategy and return summary rows.

    Each row corresponds to one held-out test city and stores key aggregate
    metrics used in model-comparison tables.
    """
    rows = []
    for test_city in city_dfs.keys():
        train_city_list = [city for city in city_dfs.keys() if city != test_city]
        train_df = pd.concat([city_dfs[city] for city in train_city_list], ignore_index=True)
        test_df = city_dfs[test_city].copy()

        x_train = train_df[model_feature_cols]
        y_train = train_df["target"]
        x_test = test_df[model_feature_cols]
        y_test = test_df["target"]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        rep_any = classification_report(
            y_test,
            y_pred,
            labels=[0, 2, 11],
            output_dict=True,
            zero_division=0,
        )
        rep = cast(dict[str, Any], rep_any)

        rows.append(
            {
                "class_weight": str(class_weight),
                "test_city": test_city,
                "macro_f1": rep["macro avg"]["f1-score"],
                "weighted_f1": rep["weighted avg"]["f1-score"],
                "recall_other": rep.get("0", {}).get("recall", np.nan),
                "recall_sidewalk": rep.get("2", {}).get("recall", np.nan),
                "recall_street": rep.get("11", {}).get("recall", np.nan),
                "n_test": len(test_df),
            }
        )
    return pd.DataFrame(rows)