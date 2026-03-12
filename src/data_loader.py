# src/data_loader.py

"""Data loading utilities for fetching and optionally processing LiDAR files.

Primary workflow:
1) download a LAZ file from remote storage (B2Drop) when needed,
2) cache it locally,
3) optionally run a lightweight PDAL processing pipeline.
"""

import os
import json
from pathlib import Path
import pdal
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables from .env file if it exists
load_dotenv()


def fetch_and_process_lidar(
    remote_filename: str,
    local_dir: str = "../data",
    filtered_suffix: str = "_filtered",
    apply_filter: bool = False,
) -> Path:
    """
    Fetch a remote LiDAR file and optionally run PDAL post-processing.

    Parameters
    ----------
    remote_filename : str
        Name of the remote file in B2Drop storage (e.g., "bologna.laz").
    local_dir : str, optional
        Local directory where files are cached.
    filtered_suffix : str, optional
        Suffix appended to the PDAL-processed output filename.
    apply_filter : bool, optional
        When True, runs a small PDAL pipeline and writes a LAS output.

    Returns
    -------
    Path
        Path to downloaded file (or processed file if `apply_filter=True`).

    Notes
    -----
    Environment variables expected:
    - DATA_DIR_FSSPEC_USER
    - DATA_DIR_FSSPEC_PASS
    - DATA_DIR_FSSPEC_BASE_URL
    - DATA_DIR_FSSPEC_URI
    """

    # --- Setup local path ---
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_raw_path = local_dir / remote_filename

    # --- Check if file already exists locally ---
    if local_raw_path.exists():
        print(f"✅ File already exists locally at {local_raw_path}, skipping download")
    else:
        # --- Setup remote B2Drop path ---
        user = os.getenv("DATA_DIR_FSSPEC_USER")
        password = os.getenv("DATA_DIR_FSSPEC_PASS")
        base_url = os.getenv("DATA_DIR_FSSPEC_BASE_URL")
        uri = os.getenv("DATA_DIR_FSSPEC_URI")
        
        # Strip webdav:// prefix from URI if present (it was for UPath, not for HTTP)
        if uri.startswith("webdav://"):
            uri = uri.replace("webdav://", "")
        
        # Construct full URL
        full_url = f"{base_url}{uri}{remote_filename}"
        print(f"🔍 DEBUG: Attempting to download from: {full_url}")
        
        # --- Read remote file using requests (simpler than UPath) ---
        try:
            response = requests.get(
                full_url, 
                auth=HTTPBasicAuth(user, password),
                timeout=300
            )
            response.raise_for_status()  # Raise exception for bad status codes
            las_bytes = response.content

            # --- Save locally ---
            with open(local_raw_path, "wb") as f:
                f.write(las_bytes)

            print(f"✅ Downloaded remote file to {local_raw_path}")
        except Exception as e:
            print(f"❌ Error downloading {remote_filename} from B2Drop: {e}")
            print(f"   Attempting to use local file if it exists...")
            if not local_raw_path.exists():
                raise FileNotFoundError(
                    f"File {remote_filename} not found in {local_dir} and could not be downloaded from B2Drop.\n"
                    f"Error details: {e}"
                )

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
