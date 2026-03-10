from __future__ import annotations

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
    return str(scale).replace(".", "p")


def _eigen_triplet_exists(available_dims: set[str], scale: float) -> bool:
    return all(
        f"{k} eigenvalue ({scale:g})" in available_dims
        for k in ("1st", "2nd", "3rd")
    )


def build_harmonized_city_df(
    las_obj,
    city_name: str,
    n_sample: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    available_dims = set(las_obj.point_format.dimension_names)

    missing_min = [col for col in MIN_REQUIRED_FIELDS if col not in available_dims]
    if missing_min:
        raise ValueError(f"{city_name} missing minimal required fields: {missing_min}")

    present_base = [col for col in BASE_COMPAT_FIELDS if col in available_dims]
    df = pd.DataFrame({col: np.asarray(getattr(las_obj, col)) for col in present_base})

    # Optional direct roughness fields when present
    for scale in EIGEN_SCALES:
        rough_col = f"Roughness ({scale:g})"
        if rough_col in available_dims:
            df[f"roughness_{_scale_suffix(scale)}"] = np.asarray(getattr(las_obj, rough_col), dtype=np.float32)

    # Derive harmonized eigen features where eigen triplets exist
    eps = 1e-12
    for scale in EIGEN_SCALES:
        if not _eigen_triplet_exists(available_dims, scale):
            continue

        l1 = np.asarray(getattr(las_obj, f"1st eigenvalue ({scale:g})"), dtype=np.float64)
        l2 = np.asarray(getattr(las_obj, f"2nd eigenvalue ({scale:g})"), dtype=np.float64)
        l3 = np.asarray(getattr(las_obj, f"3rd eigenvalue ({scale:g})"), dtype=np.float64)

        l1_safe = np.maximum(l1, eps)
        lam_sum = np.maximum(l1 + l2 + l3, eps)
        p1 = l1 / lam_sum
        p2 = l2 / lam_sum
        p3 = l3 / lam_sum
        suffix = _scale_suffix(scale)

        df[f"planarity_{suffix}"] = ((l2 - l3) / l1_safe).astype(np.float32)
        df[f"sphericity_{suffix}"] = (l3 / l1_safe).astype(np.float32)
        df[f"anisotropy_{suffix}"] = ((l1 - l3) / l1_safe).astype(np.float32)
        df[f"curvature_{suffix}"] = (l3 / lam_sum).astype(np.float32)
        df[f"eigenentropy_{suffix}"] = (
            -(p1 * np.log(np.maximum(p1, eps)) + p2 * np.log(np.maximum(p2, eps)) + p3 * np.log(np.maximum(p3, eps)))
        ).astype(np.float32)

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

    return df


def align_city_feature_space(
    city_dfs: dict[str, pd.DataFrame],
    protected_cols: tuple[str, ...] = ("target", "city", "classification"),
) -> tuple[dict[str, pd.DataFrame], list[str]]:
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