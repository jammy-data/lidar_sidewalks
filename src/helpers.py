"""Utility helpers for quick LAS metadata inspection.

This module is intentionally lightweight and notebook-friendly. The main entry
point is `describe_las`, which prints the key structural information needed
before feature engineering or model training.
"""

import laspy

def describe_las(las):
    """Print a compact summary of LAS header and spatial metadata.

    Parameters
    ----------
    las : laspy.LasData
        Loaded LAS/LAZ object.
    """
    print(f"Point Format: {las.header.point_format}")
    print(f"Number of Points: {las.header.point_count}")
    print("Available Dimensions:", list(las.point_format.dimension_names))
    print("Bounding Box:")
    print(f"  X: {las.header.mins[0]} to {las.header.maxs[0]}")
    print(f"  Y: {las.header.mins[1]} to {las.header.maxs[1]}")
    print(f"  Z: {las.header.mins[2]} to {las.header.maxs[2]}")
    print("Scale:", las.header.scales)
    print("Offset:", las.header.offsets)
    try:
        print("CRS:", las.header.parse_crs())
    except:
        print("CRS: Not defined")