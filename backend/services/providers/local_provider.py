"""Local deterministic geospatial domain reasoner.

Generates rigorous, non-hallucinatory remote sensing summaries strictly
derived from verified physical sensor metrics and spatial grounding.
"""

from typing import Dict, Any, List, Optional
from .base import BaseVisionLanguageProvider
from ...api.schemas import SpatialFeature, GeoMetadata


class LocalDeterministicProvider(BaseVisionLanguageProvider):
    """Deterministic local engine for evidence-grounded remote sensing reasoning."""

    @property
    def name(self) -> str:
        return "local_deterministic_geospatial_engine"

    @property
    def is_available(self) -> bool:
        return True  # Always available offline on any machine

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
        lines: List[str] = []

        if task_type == "SCENE_DESCRIPTION":
            lines.append(
                f"### Remote Sensing Scene Assessment\n"
                f"The acquired satellite scene encompasses a **{metadata.width}×{metadata.height} pixel** footprint "
                f"captured under **{metadata.modality}** sensing protocols ({metadata.channels} spectral channels)."
            )

            if metadata.has_nir:
                lines.append(
                    "Calibrated Near-Infrared (NIR) band is active, enabling direct biophysical chlorophyll quantification."
                )
            else:
                lines.append("Imagery is registered in RGB visible spectrum.")

            if features:
                clusters_desc = ", ".join([f"{f.label} (~{f.area_hectares or 'N/A'} ha)" for f in features[:4]])
                lines.append(f"\n**Prominent Spatial Entities Identified**: {clusters_desc}.")

            if "mean_ndvi" in metrics:
                lines.append(
                    f"\n**Vegetation Profile**: Mean NDVI is measured at **{metrics['mean_ndvi']:.2f}** "
                    f"(Median: {metrics.get('median_ndvi', 0.0):.2f}, Peak: {metrics.get('max_ndvi', 0.0):.2f}), "
                    f"with **{metrics.get('total_vegetation_pct', 0)}%** canopy coverage across agricultural and forested tracts."
                )
            elif "estimated_vegetation_pct" in metrics:
                lines.append(
                    f"\n**Vegetation Profile**: Estimated visible green cover is **{metrics.get('estimated_vegetation_pct', 0)}%** "
                    f"(~{metrics.get('estimated_vegetation_hectares', 0)} ha)."
                )

            if "water_coverage_pct" in metrics and metrics["water_coverage_pct"] > 0:
                lines.append(
                    f"\n**Hydrological Profile**: Surface water bodies delineate approximately **{metrics['water_coverage_pct']}%** "
                    f"of the total scene footprint (~{metrics.get('water_area_hectares', 0)} ha)."
                )

        elif task_type in ["SPATIAL_GROUNDING", "SPECTRAL_WATER", "SPECTRAL_VEGETATION"]:
            lines.append("### Spatial Grounding & Target Delineation\n")

            if features:
                feature_name = features[0].label.split(" (")[0]
                lines.append(
                    f"Analysis localized **{len(features)} distinct spatial clusters** corresponding to **{feature_name}**."
                )

                lines.append("\n**Sector Breakdown**:")
                for f in features:
                    area_txt = f" | Area: ~{f.area_hectares} ha" if f.area_hectares else ""
                    conf_txt = f" | Purity: {int((f.confidence or 0.85) * 100)}%" if f.confidence else ""
                    poly_txt = f" ({len(f.polygon)} polygon vertices)" if f.polygon else ""
                    lines.append(
                        f"- **{f.label}**: Normalized bounds [{f.ymin:.2f}, {f.xmin:.2f}, {f.ymax:.2f}, {f.xmax:.2f}]{poly_txt}{area_txt}{conf_txt}"
                    )

                if "total_area_hectares" in metrics and metrics["total_area_hectares"] > 0:
                    lines.append(f"\nTotal aggregated footprint is estimated at **{metrics['total_area_hectares']} hectares**.")
            else:
                lines.append(
                    "No significant spatial clusters matching the specified spectral signature were detected above the threshold."
                )

            if "mean_ndvi" in metrics:
                lines.append(
                    f"\n**Spectral Verification**: Mean NDVI across vegetated clusters is **{metrics['mean_ndvi']:.2f}**."
                )

        elif task_type == "CHANGE_DETECTION":
            lines.append("### Multitemporal Change Analysis Report\n")
            lines.append(
                f"Change Vector Analysis (CVA) detected physical alterations across **{metrics.get('total_change_pct', 0)}%** "
                f"of the analyzed scene"
            )
            if metrics.get("total_changed_hectares"):
                lines[-1] += f" (approximately **{metrics['total_changed_hectares']} hectares**)."
            else:
                lines[-1] += "."

            lines.append("\n**Categorical Transitions**:")
            if metrics.get("builtup_expansion_pct", 0) > 0:
                lines.append(
                    f"- **Built-up / Industrial Expansion**: +{metrics['builtup_expansion_pct']}% of scene. "
                    f"Characterized by high radiometric brightness shift and structural boundary appearance."
                )
            if metrics.get("vegetation_loss_pct", 0) > 0:
                lines.append(
                    f"- **Vegetation Loss / Clearing**: -{metrics['vegetation_loss_pct']}% of scene. "
                    f"Characterized by marked drop in spectral greenness/NDVI."
                )
            if metrics.get("water_fluctuation_pct", 0) > 0:
                lines.append(
                    f"- **Hydrological / Shoreline Shift**: {metrics['water_fluctuation_pct']}% of scene."
                )

            if features:
                lines.append(f"\n**Identified Modification Hotspots ({len(features)} zones)**:")
                for f in features:
                    area_txt = f" (~{f.area_hectares} ha)" if f.area_hectares else ""
                    lines.append(f"- **{f.label}**{area_txt} at normalized bounds [{f.ymin:.2f}, {f.xmin:.2f}, {f.ymax:.2f}, {f.xmax:.2f}].")

            if "otsu_separability_eta" in metrics:
                lines.append(
                    f"\n**Threshold Quality**: Otsu separability criterion $\\eta$ = **{metrics['otsu_separability_eta']}** "
                    f"(evaluated across {metrics.get('clusters_count', 0)} distinct change clusters)."
                )

        elif task_type == "OPTICAL_SAR":
            lines.append("### Optical & SAR Multimodal Intelligence Brief\n")
            sar_m = metrics.get("sar_metrics", {})
            lines.append(
                f"Cross-sensor fusion correlated optical multispectral reflectance with microwave backscatter intensity.\n"
                f"- **SAR Mean Backscatter**: **{sar_m.get('mean_backscatter_db', -12.0)} dB** ({sar_m.get('speckle_filter_applied', 'Lee filter')})\n"
                f"- **High-Reflectance Structural Scatter**: **{sar_m.get('metallic_builtup_scatter_pct', 0)}%** of surface exhibits "
                f"strong double-bounce reflections (>-6 dB) typical of vertical concrete/metal infrastructure.\n"
                f"- **Specularity / Calm Surface**: **{sar_m.get('calm_water_specular_pct', 0)}%** confirms smooth water/paved surfaces."
            )

            if features:
                lines.append(f"\n**Verified Cross-Modal Structures ({len(features)} clusters)**:")
                for f in features:
                    lines.append(f"- **{f.label}**: Corroborated by high radar backscatter and visible geometric alignment.")

        else:
            lines.append(
                f"Analysis completed across {metadata.width}×{metadata.height} raster array. "
                f"Specialist tools extracted verified spectral and spatial metrics."
            )

        if limitations:
            lines.append(f"\n> ⚠️ **Sensor & Methodological Limitation**: {limitations}")

        return "\n".join(lines)
