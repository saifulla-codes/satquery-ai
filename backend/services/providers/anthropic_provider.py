"""Anthropic Vision Provider: Claude 3.5 Sonnet remote inference."""

import os
import base64
import httpx
from typing import Dict, Any, List, Optional
from .base import BaseVisionLanguageProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class AnthropicVisionProvider(BaseVisionLanguageProvider):
    """Anthropic Claude 3.5 Sonnet remote vision-language provider."""

    @property
    def name(self) -> str:
        return "anthropic_claude_vision"

    @property
    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

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
        if not self.is_available:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        metrics_summary = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        features_summary = "\n".join([f"- {f.label} at [{f.ymin}, {f.xmin}, {f.ymax}, {f.xmax}]" for f in features[:6]])

        content: List[Dict[str, Any]] = []

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })

        content.append({
            "type": "text",
            "text": (
                f"User Query: {query}\n"
                f"Task Type: {task_type}\n"
                f"Sensor Metadata: {metadata.width}x{metadata.height}, Modality: {metadata.modality}, CRS: {metadata.crs}\n"
                f"Verified Sensor Metrics:\n{metrics_summary}\n"
                f"Spatial Features:\n{features_summary}\n"
                f"Limitations: {limitations or 'None'}\n\n"
                "Synthesize an evidence-grounded remote sensing report answering the user query. Adhere strictly to verified physical evidence."
            ),
        })

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 800,
            "system": "You are SatQuery AI, an ISRO satellite remote-sensing vision-language analyst.",
            "messages": [{"role": "user", "content": content}],
        }

        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    parts = data.get("content", [])
                    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"])
                else:
                    return f"> ⚠️ Anthropic API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return f"> ⚠️ Anthropic Vision Call Failed: {str(e)}"
