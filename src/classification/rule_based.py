from __future__ import annotations

from collections import deque

import numpy as np

from config import PipelineConfig
from src.models import LABEL_OTHER, LABEL_ROAD, LABEL_SIDEWALK, PatchFeature


def _connected_components(nodes: set[int], adjacency: dict[int, set[int]]) -> list[set[int]]:
    visited: set[int] = set()
    components: list[set[int]] = []

    for node in nodes:
        if node in visited:
            continue
        queue = deque([node])
        comp = set([node])
        visited.add(node)
        while queue:
            cur = queue.popleft()
            for nb in adjacency.get(cur, set()):
                if nb in visited or nb not in nodes:
                    continue
                visited.add(nb)
                comp.add(nb)
                queue.append(nb)
        components.append(comp)
    return components


def _axis_alignment(a: np.ndarray, b: np.ndarray) -> float:
    den = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    return float(abs(np.dot(a, b)) / den)


def classify_patches_rule_based(
    patch_features: list[PatchFeature],
    adjacency: dict[int, set[int]],
    config: PipelineConfig,
) -> np.ndarray:
    labels = np.full(len(patch_features), LABEL_OTHER, dtype=np.uint8)
    if not patch_features:
        return labels

    features_by_id = {f.patch_id: f for f in patch_features}
    large_ids = {f.patch_id for f in patch_features if f.area >= config.large_area_threshold}

    if large_ids:
        components = _connected_components(large_ids, adjacency)
        components.sort(
            key=lambda comp: (
                sum(features_by_id[c].area for c in comp),
                -np.mean([features_by_id[c].z_mean for c in comp]),
            ),
            reverse=True,
        )
        road_ids = components[0]
    else:
        road_id = min(patch_features, key=lambda f: (f.z_mean, -f.area)).patch_id
        road_ids = {road_id}

    for rid in road_ids:
        labels[rid] = LABEL_ROAD

    road_axes = [features_by_id[r].major_axis_xy for r in road_ids]
    road_z_ref = float(np.median([features_by_id[r].z_mean for r in road_ids]))
    relax = max(config.sidewalk_relax_factor, 1.0)

    for road_id in road_ids:
        road_feat = features_by_id[road_id]
        for nb in adjacency.get(road_id, set()):
            if labels[nb] == LABEL_ROAD:
                continue
            cand = features_by_id[nb]
            dz = cand.z_mean - road_feat.z_mean
            dz_global = cand.z_mean - road_z_ref

            max_alignment = 0.0
            if road_axes:
                max_alignment = max(_axis_alignment(cand.major_axis_xy, ax) for ax in road_axes)

            curb_case = config.curb_height_min < abs(dz) < config.curb_height_max and dz > -0.03
            no_curb_case = (
                -0.03 < dz_global < config.road_relative_height_max
                and cand.roughness <= config.sidewalk_roughness_max * relax
                and cand.mean_curvature <= config.sidewalk_curvature_max * relax
                and (cand.elongation >= config.sidewalk_elongation_min / relax or cand.area >= config.sidewalk_seed_min_area)
                and max_alignment >= config.sidewalk_alignment_min / relax
            )
            grass_like = (
                cand.roughness >= config.grass_roughness_min * relax
                or cand.mean_curvature >= config.grass_curvature_min * relax
                or cand.area <= config.tiny_patch_area
            )

            if (curb_case or no_curb_case) and not grass_like:
                labels[nb] = LABEL_SIDEWALK

    sidewalk_queue = deque(int(i) for i in np.flatnonzero(labels == LABEL_SIDEWALK))
    patch_hops: dict[int, int] = {int(i): 0 for i in np.flatnonzero(labels == LABEL_SIDEWALK)}

    while sidewalk_queue:
        sid = sidewalk_queue.popleft()
        hop = patch_hops.get(sid, 0)
        if hop >= max(int(config.sidewalk_expand_hops), 0):
            continue

        for nb in adjacency.get(sid, set()):
            if labels[nb] != LABEL_OTHER:
                continue
            feat = features_by_id[nb]
            dz_global = feat.z_mean - road_z_ref
            grass_like = (
                feat.roughness >= config.grass_roughness_min * relax
                or feat.mean_curvature >= config.grass_curvature_min * relax
                or feat.area <= config.tiny_patch_area
            )
            if grass_like:
                continue

            if -0.03 < dz_global < config.road_relative_height_max and feat.roughness <= config.sidewalk_roughness_max * relax and feat.mean_curvature <= config.sidewalk_curvature_max * relax:
                labels[nb] = LABEL_SIDEWALK
                patch_hops[nb] = hop + 1
                sidewalk_queue.append(nb)

    return labels
