"""Analysis Service: Manages in-memory and persistent storage of past analysis jobs."""

import os
import json
from typing import Dict, Any, Optional, List
from ..api.schemas import AnalysisRecord


class AnalysisService:
    """Service for caching and querying analysis runs."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_dir: str = ""

    @classmethod
    def init(cls, cache_dir: str):
        cls._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Load existing cached analyses
        for fname in os.listdir(cache_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(cache_dir, fname), "r") as f:
                        data = json.load(f)
                        cls._cache[data["analysis_id"]] = data
                except Exception:
                    pass

    @classmethod
    def save_analysis(cls, analysis_data: Dict[str, Any]):
        aid = analysis_data.get("analysis_id")
        if not aid:
            return
        cls._cache[aid] = analysis_data
        if cls._cache_dir:
            file_path = os.path.join(cls._cache_dir, f"{aid}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(analysis_data, f, indent=2)
            except Exception:
                pass

    @classmethod
    def get_analysis(cls, analysis_id: str) -> Optional[Dict[str, Any]]:
        return cls._cache.get(analysis_id)

    @classmethod
    def list_recent(cls, limit: int = 20) -> List[Dict[str, Any]]:
        items = list(cls._cache.values())
        items.sort(key=lambda x: x.get("execution_time_ms", 0), reverse=True)
        return items[:limit]
