"""VRSBench Dataset Adapter for SatQuery AI.

Standardizes high-resolution optical Vision-Language benchmark annotations
including detailed remote sensing captions, visual grounding bounding boxes, and VQA pairs.
"""

import os
import json
from typing import Dict, Any, List, Optional
from ..base_adapter import BaseDatasetAdapter


class VRSBenchAdapter(BaseDatasetAdapter):
    """Adapter for VRSBench high-resolution remote sensing vision-language benchmark."""

    def __init__(self, root_dir: Optional[str] = None, annotations_file: Optional[str] = None):
        super().__init__(root_dir)
        self.annotations_file = annotations_file or (
            os.path.join(self.root_dir, "vrsbench_annotations.json") if self.root_dir else None
        )
        self._data: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "VRSBench"

    @property
    def dataset_type(self) -> str:
        return "High-Resolution Optical Remote Sensing VQA & Grounding"

    def _default_env_root(self) -> Optional[str]:
        return os.getenv("DATASET_ROOT_VRSBENCH")

    @property
    def is_available(self) -> bool:
        if self.annotations_file and os.path.exists(self.annotations_file):
            return True
        return super().is_available

    def _load_annotations(self):
        if self._data is None:
            if self.annotations_file and os.path.exists(self.annotations_file):
                with open(self.annotations_file, "r") as f:
                    self._data = json.load(f)
            else:
                self._data = {}

    def list_samples(self) -> List[str]:
        if not self.is_available:
            return []
        self._load_annotations()
        return list(self._data.keys()) if self._data else []

    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        """Loads caption, visual question answering pairs, and object bounding boxes."""
        if not self.is_available:
            raise FileNotFoundError(
                "VRSBench annotations not found. Mount benchmark and configure DATASET_ROOT_VRSBENCH."
            )
        self._load_annotations()
        if self._data and sample_id in self._data:
            return self._data[sample_id]
        raise KeyError(f"Sample ID '{sample_id}' not found in VRSBench annotations.")

    # Backward compatibility alias
    get_sample_entry = load_sample
