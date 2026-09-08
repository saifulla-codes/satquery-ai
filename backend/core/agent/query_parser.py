"""Query Understanding & Intent Classification for remote sensing queries."""

import re
from typing import Dict, Any, List


class QueryIntent:
    SCENE_DESCRIPTION = "SCENE_DESCRIPTION"
    SPATIAL_GROUNDING = "SPATIAL_GROUNDING"
    SPECTRAL_VEGETATION = "SPECTRAL_VEGETATION"
    SPECTRAL_WATER = "SPECTRAL_WATER"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    COMPOUND_CHANGE = "COMPOUND_CHANGE"
    OPTICAL_SAR = "OPTICAL_SAR"
    LAND_COVER = "LAND_COVER"


class QueryParser:
    """Parses natural-language satellite queries into structured operational intents."""

    @classmethod
    def parse(cls, query: str, context_modality: str = "OPTICAL") -> Dict[str, Any]:
        q = query.lower().strip()

        # Intent detection heuristics
        intents: List[str] = []
        target_features: List[str] = []

        # Compound change detection (e.g. vegetation decreased while built-up increased)
        has_temporal = any(w in q for w in ["change", "between", "earlier", "later", "before", "after", "two dates", "evolution", "decreased", "increased"])
        has_veg = any(w in q for w in ["vegetation", "crop", "forest", "farm", "green", "tree", "plant", "ndvi"])
        has_builtup = any(w in q for w in ["built", "urban", "construction", "development", "building", "structure"])

        if has_temporal and has_veg and has_builtup:
            intents.append(QueryIntent.COMPOUND_CHANGE)
            intents.append(QueryIntent.CHANGE_DETECTION)
            target_features.extend(["vegetation_loss", "builtup_gain"])
        elif has_temporal:
            intents.append(QueryIntent.CHANGE_DETECTION)

        # SAR / Radar patterns
        if any(w in q for w in ["sar", "radar", "backscatter", "microwave", "sentinel-1", "dielectric"]):
            intents.append(QueryIntent.OPTICAL_SAR)

        # Spatial Grounding ("Where is...", "Find...", "Show me...", "Locate...", "Detect...")
        if any(w in q for w in ["where", "find", "locate", "show", "detect", "highlight", "identify", "pinpoint"]):
            intents.append(QueryIntent.SPATIAL_GROUNDING)

        # Spectral Vegetation / NDVI
        if any(w in q for w in ["vegetation", "crop", "forest", "farm", "green", "ndvi", "agricultural", "tree", "plant"]):
            intents.append(QueryIntent.SPECTRAL_VEGETATION)
            target_features.append("vegetation")

        # Spectral Water / NDWI
        if any(w in q for w in ["water", "river", "lake", "ocean", "sea", "canal", "reservoir", "ndwi", "flood", "bay", "coastal"]):
            intents.append(QueryIntent.SPECTRAL_WATER)
            target_features.append("water")

        # Built-up / Urban / Infrastructure
        if any(w in q for w in ["urban", "city", "building", "structure", "built-up", "construction", "road", "port", "facility"]):
            target_features.append("built_up")

        # Scene Description
        if any(w in q for w in ["describe", "what do you see", "what is visible", "overview", "caption", "tell me about", "what is in"]):
            intents.append(QueryIntent.SCENE_DESCRIPTION)

        # Land cover query
        if any(w in q for w in ["land cover", "land use", "types of land", "classes"]):
            intents.append(QueryIntent.LAND_COVER)

        # Default fallback intent
        if not intents:
            if target_features:
                intents.append(QueryIntent.SPATIAL_GROUNDING)
            else:
                intents.append(QueryIntent.SCENE_DESCRIPTION)

        primary_intent = intents[0]

        return {
            "query": query,
            "primary_intent": primary_intent,
            "intents": intents,
            "target_features": target_features,
            "requires_grounding": QueryIntent.SPATIAL_GROUNDING in intents or len(target_features) > 0,
            "requires_spectral": QueryIntent.SPECTRAL_VEGETATION in intents or QueryIntent.SPECTRAL_WATER in intents,
            "requires_change": QueryIntent.CHANGE_DETECTION in intents,
            "requires_sar": QueryIntent.OPTICAL_SAR in intents or context_modality == "SAR",
        }
