"""Google Gemini Vision Provider: Gemini 1.5 Flash / Pro remote inference."""

import os
import base64
import httpx
from typing import Dict, Any, List, Optional
from .base import BaseVisionLanguageProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class GeminiVisionProvider(BaseVisionLanguageProvider):
    """Google Gemini remote vision-language provider."""

    @property
    def name(self) -> str:
        return "google_gemini_vision"

    @property
    def is_available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

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
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        metrics_summary = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        features_summary = "\n".join([f"- {f.label} at [{f.ymin}, {f.xmin}, {f.ymax}, {f.xmax}]" for f in features[:6]])

        parts: List[Dict[str, Any]] = [
            {
                "text": (
                    "You are SatQuery AI, an ISRO remote-sensing specialist. "
                    "Analyze the satellite scene conditioned strictly on the provided verified physical metrics.\n\n"
                    f"User Query: {query}\n"
                    f"Task Type: {task_type}\n"
                    f"Image Metadata: {metadata.width}x{metadata.height}, Modality: {metadata.modality}, CRS: {metadata.crs}\n"
                    f"Verified Sensor Metrics:\n{metrics_summary}\n"
                    f"Spatial Features:\n{features_summary}\n"
                    f"Limitations: {limitations or 'None'}\n\n"
                    "Provide a professional, grounded assessment answering the query."
                )
            }
        ]

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64,
                    }
                })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        return candidates[0]["content"]["parts"][0]["text"]
                return f"> ⚠️ Gemini API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return f"> ⚠️ Gemini Request Failed: {str(e)}"
