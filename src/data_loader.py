# src/data_loader.py

import os
import json
from pathlib import Path
from upath import UPath
import pdal
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()


def fetch_and_process_lidar(
    remote_filename: str,
    local_dir: str = "../data",
    filtered_suffix: str = "_filtered",
    apply_filter: bool = True,
) -> Path:
    """
    Fetches a remote LiDAR (.laz) file from B2Drop, saves it locally, 
    and optionally processes it with PDAL.

    Parameters
    ----------
    remote_filename : str
        Name of the remote file in your B2Drop storage (e.g., "bologna.laz").
    local_dir : str, optional
        Local directory where the file will be saved (default: "../data").
    filtered_suffix : str, optional
        Suffix to append to the processed file name (default: "_filtered").
    apply_filter : bool, optional
        If True, runs a PDAL pipeline with a basic stats filter.

    Returns
    -------
    Path
        Path to the processed (or downloaded) LAS file.
    """

    # --- Setup remote B2Drop path ---
    B2D_DIR = UPath(
        os.getenv("DATA_DIR_FSSPEC_URI"),
        base_url=os.getenv("DATA_DIR_FSSPEC_BASE_URL"),
        auth=(
            os.getenv("DATA_DIR_FSSPEC_USER"),
            os.getenv("DATA_DIR_FSSPEC_PASS"),
        ),
    )

    file_path = B2D_DIR / remote_filename
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # --- Read remote file as binary stream ---
    local_raw_path = local_dir / remote_filename
    with file_path.open("rb") as f:
        las_bytes = f.read()

    # --- Save locally ---
    with open(local_raw_path, "wb") as f:
        f.write(las_bytes)

    print(f"✅ Downloaded remote file to {local_raw_path}")

    # --- Optionally process with PDAL ---
    if apply_filter:
        local_filtered_path = local_dir / f"{local_raw_path.stem}{filtered_suffix}.las"

        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(local_raw_path)},
                {"type": "filters.stats"},
                {"type": "writers.las", "filename": str(local_filtered_path)},
            ]
        }

        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        print(f"✅ PDAL processed file saved at {local_filtered_path}")
        return local_filtered_path

    return local_raw_path
