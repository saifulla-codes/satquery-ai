"""Dataset adapters package for SatQuery AI remote sensing benchmarks."""

from .base_adapter import BaseDatasetAdapter
from .bigearthnet.adapter import BigEarthNetAdapter
from .vrsbench.adapter import VRSBenchAdapter
from .rsvqa.adapter import RSVQAAdapter, CDVQAAdapter

__all__ = [
    "BaseDatasetAdapter",
    "BigEarthNetAdapter",
    "VRSBenchAdapter",
    "RSVQAAdapter",
    "CDVQAAdapter",
]
