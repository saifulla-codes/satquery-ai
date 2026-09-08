"""OpenAI Vision Provider: GPT-4o / GPT-4o-mini remote inference."""

import os
import base64
import httpx
from typing import Dict, Any, List, Optional
from .base import BaseVisionLanguageProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class OpenAIVisionProvider(BaseVisionLanguageProvider):
    """OpenAI GPT-4o remote vision-language provider."""

    @property
    def name(self) -> str:
        return "openai_gpt4o_vision"

    @property
    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

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
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        api_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        metrics_summary = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        features_summary = "\n".join([f"- {f.label} at [{f.ymin}, {f.xmin}, {f.ymax}, {f.xmax}]" for f in features[:6]])

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"User Query: {query}\n"
                    f"Task Type: {task_type}\n"
                    f"Sensor Metadata: {metadata.width}x{metadata.height}, Modality: {metadata.modality}, CRS: {metadata.crs}\n"
                    f"Verified Sensor Metrics:\n{metrics_summary}\n"
                    f"Spatial Features:\n{features_summary}\n"
                    f"Limitations: {limitations or 'None'}\n\n"
                    "Synthesize an evidence-grounded remote sensing report answering the query."
                ),
            }
        ]

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are SatQuery AI, an ISRO satellite remote-sensing analyst. Answer based strictly on physical data.",
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": 800,
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                return f"> ⚠️ OpenAI API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return f"> ⚠️ OpenAI Request Failed: {str(e)}"
