"""Vision-Language reasoning service wrapping ProviderRouter."""

from typing import Dict, Any, List, Optional, Tuple
from .providers.provider_router import ProviderRouter
from ..api.schemas import SpatialFeature, GeoMetadata


class LLMService:
    """Service facade providing evidence-grounded natural-language synthesis."""

    @classmethod
    def synthesize_answer(
        cls,
        query: str,
        task_type: str,
        metadata: GeoMetadata,
        metrics: Dict[str, Any],
        features: List[SpatialFeature],
        limitations: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Synthesizes answer and returns (answer, provider_name, analysis_mode)."""
        return ProviderRouter.synthesize(
            query=query,
            task_type=task_type,
            metadata=metadata,
            metrics=metrics,
            features=features,
            limitations=limitations,
            image_path=image_path,
        )
