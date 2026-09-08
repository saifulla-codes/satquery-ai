"""Clean Provider Abstractions for SatQuery AI: VisionLanguageProvider, LanguageReasoningProvider, and GroundingProvider."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from ...api.schemas import SpatialFeature, GeoMetadata


class VisionLanguageProvider(ABC):
    """Abstract base provider for multimodal vision-language synthesis."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier, e.g. 'local_deterministic_geospatial_engine', 'groq_llama_3_2_vision'."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the required credentials/endpoints are active."""
        pass

    @property
    def provider_type(self) -> str:
        """Returns 'LOCAL ANALYSIS' or 'REMOTE AI ANALYSIS'."""
        return "LOCAL ANALYSIS" if "local" in self.name else f"REMOTE AI ANALYSIS ({self.name})"

    @abstractmethod
    def synthesize_response(
        self,
        query: str,
        task_type: str,
        metadata: GeoMetadata,
        metrics: Dict[str, Any],
        features: List[SpatialFeature],
        limitations: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """Generates an evidence-grounded vision-language response."""
        pass


# Backwards compatibility alias
BaseVisionLanguageProvider = VisionLanguageProvider


class LanguageReasoningProvider(ABC):
    """Abstract base provider for natural language query understanding and task planning."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def decompose_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decomposes a query into operational intents, target modalities, and temporal conditions."""
        pass


class GroundingProvider(ABC):
    """Abstract base provider for spatial grounding and object localization in pixel space."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def ground(
        self,
        raster: Any,
        feature_type: str,
        threshold_override: Optional[float] = None,
    ) -> Tuple[List[SpatialFeature], np.ndarray, Dict[str, Any]]:
        """Grounds specified target features and extracts bounding geometry."""
        pass
