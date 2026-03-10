# lidar_sidewalks

Geometry-first pipeline to classify TLS point clouds into:
- `0` = other
- `1` = road
- `2` = sidewalk

The baseline is rule-based (no deep learning), patch-centric, and designed to scale to large clouds through downsampling and configurable parallel computation.

## Implemented Pipeline

1. Preprocessing
	- Optional voxel downsampling
	- kNN + PCA normal estimation
	- Curvature and roughness estimation
2. Horizontal extraction
	- Filters by vertical normal component and curvature (optional roughness)
3. Superpoint segmentation
	- Region growing on horizontal candidates
4. Patch features
	- Height stats, roughness, curvature, area proxy, principal axes, elongation
5. Patch graph
	- Bounding-box proximity graph
6. Rule-based classification
	- Road identification from large/connected horizontal patches
	- Sidewalk identification for curb and no-curb cases
	- Grass rejection based on roughness/curvature/area
7. Postprocess
	- Small fragment cleanup
	- Optional point-level kNN majority smoothing
8. Optional ML boost
	- RandomForest helpers for patch-level training/inference

## Source Layout

- `src/geometry/preprocess.py`
- `src/geometry/normals.py`
- `src/segmentation/horizontal_filter.py`
- `src/segmentation/region_growing.py`
- `src/graph/adjacency.py`
- `src/classification/rule_based.py`
- `src/classification/random_forest.py`
- `src/postprocess/smoothing.py`
- `src/pipeline.py`
- `config.py`

## Usage

```python
import numpy as np
from config import PipelineConfig
from src.pipeline import LidarRoadSidewalkPipeline

points_xyz = np.random.rand(1_000_000, 3)  # replace with TLS cloud XYZ

config = PipelineConfig(
	 voxel_size=0.1,
	 k_neighbors=30,
	 normal_vertical_threshold=0.9,
	 curvature_threshold=0.05,
	 region_normal_angle_deg=10.0,
	 region_distance=0.2,
	 curb_height_min=0.05,
	 curb_height_max=0.25,
	 large_area_threshold=10.0,
	segmentation_mode="region_growing",  # or "stripe_octree"
)

pipeline = LidarRoadSidewalkPipeline(config)
labels = pipeline.run(points_xyz)  # shape (N,), values in {0,1,2}
```

To inspect internal outputs:

```python
labels, aux = pipeline.run(points_xyz, return_intermediate=True)
```

## Clear Step-by-Step Runbook

### Step 1 — Load point cloud as `Nx3`

```python
import laspy
import numpy as np

las = laspy.read("/path/to/cloud.laz")
points_xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
```

### Step 2 — Configure thresholds

```python
from config import PipelineConfig

config = PipelineConfig(
	voxel_size=0.1,
	k_neighbors=30,
	normal_vertical_threshold=0.9,
	curvature_threshold=0.05,
	region_normal_angle_deg=10.0,
	region_distance=0.2,
	curb_height_min=0.05,
	curb_height_max=0.25,
	large_area_threshold=10.0,
)
```

### Step 3 — Run baseline pipeline

```python
from src.pipeline import LidarRoadSidewalkPipeline

pipeline = LidarRoadSidewalkPipeline(config)
labels = pipeline.run(points_xyz)
```

### Step 4 — Inspect class counts

```python
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
```

### Step 5 — Save labels (optional)

```python
las.add_extra_dim(laspy.ExtraBytesParams(name="pred_label", type=np.uint8))
las.pred_label = labels
las.write("/path/to/cloud_with_labels.laz")
```

### Step 6 — Large cloud mode (tiled)

Use this for clouds where full-scene neighborhood processing is heavy.

```python
from src.pipeline_tiled import run_pipeline_tiled

labels_tiled = run_pipeline_tiled(
	points_xyz=points_xyz,
	pipeline=pipeline,
	tile_size_xy=100.0,
	tile_overlap_xy=2.0,
	min_points_per_tile=500,
)
```

### Step 6b — Stripe + Octree mode (long-distance distortion handling)

Use this when terrain slope or long-range drift makes global geometry less stable.

```python
config_octree = PipelineConfig(
	segmentation_mode="stripe_octree",
	stripe_width=25.0,
	octree_max_leaf_points=400,
	octree_min_cell_size=0.5,
	octree_planarity_max_curvature=0.02,
	octree_merge_distance=1.0,
	octree_merge_normal_angle_deg=10.0,
	octree_merge_height_delta=0.08,
)

pipeline_octree = LidarRoadSidewalkPipeline(config_octree)
labels_octree = pipeline_octree.run(points_xyz)
```

### Step 7 — Debug intermediate outputs (optional)

```python
labels_dbg, aux = pipeline.run(points_xyz, return_intermediate=True)
print(aux.keys())
```

### Step 8 — Automatic parameter sweep (recommended)

Because sidewalk geometry can vary by street (narrow sidewalks, drive-over ramps,
wide sidewalks, local slope), one global threshold set is often suboptimal.

Recommended workflow:

1. Define a small candidate grid for sensitive parameters:
	- `curb_height_min`
	- `curb_height_max`
	- `sidewalk_roughness_max`
	- `sidewalk_alignment_min`
	- for stripe mode: `octree_merge_height_delta`
2. Evaluate each candidate set against ground truth labels (IoU road + IoU sidewalk).
3. Report both:
	- global score (whole cloud), and
	- local scores by spatial zones (e.g., X-axis bins) to detect street-specific failures.
4. Choose either:
	- one robust global config (best average), or
	- zone-specific configs (best per zone) for heterogeneous areas.

The notebook `Notebooks/5_DL_exploration.ipynb` includes a sweep section with this logic.

## Notes on Scaling

- Use downsampling (`voxel_size=0.05` to `0.1`) for very large clouds.
- `kNN` normal estimation uses parallel workers (`n_jobs` in config).
- For >10M points, use tiled mode (`run_pipeline_tiled`) and tune:
	- `tile_size_xy`: 80–200 m
	- `tile_overlap_xy`: 1–3 m
	- `tile_min_points`: skip sparse tiles
- Expect parameter drift across neighborhoods: evaluate by spatial zones before finalizing defaults.

## Runtime Hotspots (Notebook)

- Baseline full-cloud run can take minutes on large point sets.
- Tiled mode is often slower but safer for memory at very large scale.
- Two-method comparison roughly doubles runtime.
- Parameter sweep is the heaviest stage because it reruns the full pipeline many times.

Remote server usage recommendations:
- Keep `n_jobs=-1` on dedicated servers to use all CPU cores.
- Start with a reduced sweep grid, then expand around best candidates.
- Use tiled mode for very large clouds to prevent memory pressure spikes.


