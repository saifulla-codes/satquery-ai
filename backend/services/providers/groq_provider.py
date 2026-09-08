"""Groq Vision Provider: Llama-3.2 Vision remote inference."""

import os
import base64
from typing import Dict, Any, List, Optional
from .base import BaseVisionLanguageProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class GroqVisionProvider(BaseVisionLanguageProvider):
    """Groq-hosted Llama-3.2-11b-vision-preview provider."""

    @property
    def name(self) -> str:
        return "groq_llama_3_2_vision"

    @property
    def is_available(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

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
            raise RuntimeError("GROQ_API_KEY is not configured.")

        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            # Build grounded prompt containing exact verified metrics
            system_prompt = (
                "You are SatQuery AI, an expert satellite imagery and remote sensing analyst for ISRO. "
                "You are provided with real quantitative sensor metrics, spectral indices, and spatial detections. "
                "You must strictly adhere to the computed physical data provided and never hallucinate false features."
            )

            metrics_summary = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
            features_summary = "\n".join([f"- {f.label} at [{f.ymin}, {f.xmin}, {f.ymax}, {f.xmax}]" for f in features[:6]])

            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"User Query: {query}\n"
                        f"Task Type: {task_type}\n"
                        f"Image Metadata: {metadata.width}x{metadata.height}, {metadata.channels} channels, Modality: {metadata.modality}, CRS: {metadata.crs}\n"
                        f"Verified Sensor Metrics:\n{metrics_summary}\n"
                        f"Spatial Grounding Features:\n{features_summary}\n"
                        f"Sensor Limitations: {limitations or 'None'}\n\n"
                        "Provide a structured, professional satellite intelligence brief answering the user query."
                    ),
                }
            ]

            # If image_path provided and exists, encode base64
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model="llama-3.2-11b-vision-preview",
                temperature=0.2,
                max_tokens=800,
            )

            return chat_completion.choices[0].message.content or "No response received from Groq."
        except Exception as e:
            return f"> ⚠️ Groq VLM Call Failed: {str(e)}\n\n(Fallback to local deterministic reasoning required)."
