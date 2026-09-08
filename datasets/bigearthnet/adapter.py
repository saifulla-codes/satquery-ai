"""BigEarthNet Dataset Adapter for SatQuery AI.

Handles Sentinel-2 12-band and Sentinel-1 dual-polarization BigEarthNet patch loading,
multi-label Corine Land Cover classification mapping, and benchmark evaluation.
"""

import os
import json
from typing import Dict, Any, List, Optional
from ..base_adapter import BaseDatasetAdapter


class BigEarthNetAdapter(BaseDatasetAdapter):
    """Adapter for loading BigEarthNet benchmark satellite patches."""

    CORINE_LAND_COVER_CLASSES = [
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Industrial or commercial units",
        "Road and rail networks",
        "Port areas",
        "Airports",
        "Non-irrigated arable land",
        "Permanently irrigated land",
        "Rice fields",
        "Vineyards",
        "Fruit trees and berry plantations",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland",
        "Moors and heathland",
        "Sclerophyllous vegetation",
        "Transitional woodland/shrub",
        "Beaches, dunes, sands",
        "Bare rock",
        "Sparsely vegetated areas",
        "Burnt areas",
        "Inland marshes",
        "Peat bogs",
        "Salt marshes",
        "Salines",
        "Intertidal flats",
        "Water courses",
        "Water bodies",
        "Coastal lagoons",
        "Estuaries",
        "Sea and ocean",
    ]

    @property
    def name(self) -> str:
        return "BigEarthNet"

    @property
    def dataset_type(self) -> str:
        return "Sentinel-2 / Sentinel-1 Multi-Label Land Cover"

    def _default_env_root(self) -> Optional[str]:
        return os.getenv("DATASET_ROOT_BIGEARTHNET")

    def list_samples(self) -> List[str]:
        if not self.is_available or not self.root_dir:
            return []
        return [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith(".")
        ]

    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        """Loads a BigEarthNet patch directory containing GeoTIFF bands and JSON metadata."""
        if not self.is_available or not self.root_dir:
            raise FileNotFoundError(f"BigEarthNet dataset directory not mounted. Configure DATASET_ROOT_BIGEARTHNET.")

        patch_dir = os.path.join(self.root_dir, sample_id)
        if not os.path.exists(patch_dir):
            raise FileNotFoundError(f"BigEarthNet patch not found: {patch_dir}")

        json_path = os.path.join(patch_dir, f"{sample_id}_labels_metadata.json")
        labels = []
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                labels = data.get("labels", [])

        band_files = [f for f in os.listdir(patch_dir) if f.endswith(".tif")]

        return {
            "patch_name": sample_id,
            "labels": labels,
            "corine_classes": [cls for cls in labels if cls in self.CORINE_LAND_COVER_CLASSES],
            "bands_available": band_files,
            "band_count": len(band_files),
        }

    # Backward compatibility alias
    load_patch = load_sample
