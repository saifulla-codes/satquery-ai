"""Provider Router: Selects and delegates to active Vision-Language or Deterministic Reasoner."""

import os
from typing import Dict, Any, List, Optional, Tuple
from .base import VisionLanguageProvider, BaseVisionLanguageProvider
from .local_provider import LocalDeterministicProvider
from .groq_provider import GroqVisionProvider
from .openai_provider import OpenAIVisionProvider
from .gemini_provider import GeminiVisionProvider
from .anthropic_provider import AnthropicVisionProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class ProviderRouter:
    """Manages AI model provider registry and routes synthesis requests."""

    _providers: Dict[str, VisionLanguageProvider] = {}
    _local_fallback: VisionLanguageProvider = LocalDeterministicProvider()

    @classmethod
    def initialize(cls):
        cls.register(LocalDeterministicProvider())
        cls.register(GroqVisionProvider())
        cls.register(OpenAIVisionProvider())
        cls.register(GeminiVisionProvider())
        cls.register(AnthropicVisionProvider())

    @classmethod
    def register(cls, provider: VisionLanguageProvider):
        cls._providers[provider.name] = provider

    @classmethod
    def get_active_provider(cls) -> VisionLanguageProvider:
        """Determines the active provider based on environment configuration."""
        configured_pref = os.getenv("ACTIVE_LLM_PROVIDER", "").lower().strip()

        # If explicitly requested and available
        if configured_pref in cls._providers and cls._providers[configured_pref].is_available:
            return cls._providers[configured_pref]

        # Auto-detect available external providers in priority order
        if os.getenv("GROQ_API_KEY"):
            return cls._providers.get("groq_llama_3_2_vision", cls._local_fallback)
        if os.getenv("ANTHROPIC_API_KEY"):
            return cls._providers.get("anthropic_claude_vision", cls._local_fallback)
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return cls._providers.get("google_gemini_vision", cls._local_fallback)
        if os.getenv("OPENAI_API_KEY"):
            return cls._providers.get("openai_gpt4o_vision", cls._local_fallback)

        # Default to local deterministic engine
        return cls._local_fallback

    @classmethod
    def synthesize(
        cls,
        query: str,
        task_type: str,
        metadata: GeoMetadata,
        metrics: Dict[str, Any],
        features: List[SpatialFeature],
        limitations: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Synthesizes response and returns (answer_text, provider_name, analysis_mode)."""
        provider = cls.get_active_provider()
        mode = provider.provider_type
        try:
            answer = provider.synthesize_response(
                query=query,
                task_type=task_type,
                metadata=metadata,
                metrics=metrics,
                features=features,
                limitations=limitations,
                image_path=image_path,
            )
            return answer, provider.name, mode
        except Exception as e:
            # Fallback to deterministic local provider on failure
            answer = cls._local_fallback.synthesize_response(
                query=query,
                task_type=task_type,
                metadata=metadata,
                metrics=metrics,
                features=features,
                limitations=f"{limitations or ''} [External VLM failed: {str(e)}; fell back to local deterministic reasoner]".strip(),
                image_path=image_path,
            )
            return answer, cls._local_fallback.name, "LOCAL ANALYSIS (Fallback)"

    @classmethod
    def list_providers(cls) -> List[Dict[str, Any]]:
        active = cls.get_active_provider()
        return [
            {
                "name": p.name,
                "type": "Local Deterministic" if "local" in p.name else "Cloud / Remote VLM",
                "mode": p.provider_type,
                "available": p.is_available,
                "is_active": p.name == active.name,
            }
            for p in cls._providers.values()
        ]


ProviderRouter.initialize()
