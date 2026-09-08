"""Base dataset adapter interface for remote-sensing benchmark datasets in SatQuery AI."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseDatasetAdapter(ABC):
    """Abstract base adapter for remote sensing vision-language and classification datasets."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or self._default_env_root()

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name, e.g. 'BigEarthNet', 'VRSBench', 'RSVQA', 'CDVQA'."""
        pass

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Modality or task type, e.g. 'Multispectral Land Cover', 'Change VQA', etc."""
        pass

    @property
    def is_available(self) -> bool:
        """Returns True if the dataset directory is mounted and exists locally."""
        return bool(self.root_dir and os.path.exists(self.root_dir))

    @abstractmethod
    def _default_env_root(self) -> Optional[str]:
        """Reads environment variable for default dataset directory."""
        pass

    @abstractmethod
    def list_samples(self) -> List[str]:
        """Lists available sample IDs within the dataset."""
        pass

    @abstractmethod
    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        """Loads annotations, imagery paths, and metadata for a specific sample."""
        pass
