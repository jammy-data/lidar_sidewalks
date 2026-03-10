from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
	voxel_size: float = 0.1
	use_downsample: bool = True
	k_neighbors: int = 30
	n_jobs: int = -1
	normal_chunk_size: int = 200_000

	normal_vertical_threshold: float = 0.9
	curvature_threshold: float = 0.05
	roughness_threshold: float | None = None

	region_normal_angle_deg: float = 10.0
	region_distance: float = 0.2
	region_curvature_delta: float = 0.02
	region_height_delta: float = 0.06
	min_patch_points: int = 20
	segmentation_mode: str = "region_growing"

	stripe_width: float = 25.0
	octree_max_leaf_points: int = 400
	octree_min_cell_size: float = 0.5
	octree_planarity_max_curvature: float = 0.02
	octree_merge_distance: float = 1.0
	octree_merge_normal_angle_deg: float = 10.0
	octree_merge_height_delta: float = 0.08

	adjacency_distance: float = 0.3

	curb_height_min: float = 0.05
	curb_height_max: float = 0.25
	large_area_threshold: float = 10.0

	sidewalk_roughness_max: float = 0.03
	sidewalk_curvature_max: float = 0.05
	sidewalk_elongation_min: float = 1.5
	sidewalk_alignment_min: float = 0.6
	road_relative_height_max: float = 0.35
	sidewalk_relax_factor: float = 1.4
	sidewalk_expand_hops: int = 2
	sidewalk_seed_min_area: float = 1.0

	grass_roughness_min: float = 0.05
	grass_curvature_min: float = 0.08
	tiny_patch_area: float = 0.5

	merge_small_sidewalk_area: float = 1.0
	remove_isolated_area: float = 0.3
	smooth_knn_k: int = 0

	enable_rf_refinement: bool = False

	tile_size_xy: float = 100.0
	tile_overlap_xy: float = 2.0
	tile_min_points: int = 500
	tile_n_jobs: int = -1

