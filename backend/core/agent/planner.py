"""Task Planner: Dynamically constructs the execution DAG of specialist remote sensing tools."""

from typing import List, Dict, Any
from ...api.schemas import ExecutionTraceStep
from .query_parser import QueryIntent


class TaskPlanner:
    """Plans multi-step execution graphs for remote sensing queries."""

    @classmethod
    def plan(
        cls,
        parsed_query: Dict[str, Any],
        has_nir: bool = False,
        is_multitemporal: bool = False,
        is_optical_sar: bool = False,
    ) -> List[ExecutionTraceStep]:
        steps: List[ExecutionTraceStep] = []

        # 1. Query Understanding Step
        steps.append(
            ExecutionTraceStep(
                step_id="step_query_parse",
                step_name="Query Understanding & Intent Decomposition",
                tool_name="query_parser",
                status="completed",
                summary=f"Decomposed intent: {parsed_query['primary_intent']}",
                inputs_summary={"query": parsed_query["query"]},
            )
        )

        # 2. Raster Decoding Step
        steps.append(
            ExecutionTraceStep(
                step_id="step_raster_decode",
                step_name="Raster & GeoTIFF Metadata Extraction",
                tool_name="geotiff_decoder",
                status="pending",
                summary="Extracting spatial CRS, pixel dimensions, and spectral band allocations",
            )
        )

        primary_intent = parsed_query["primary_intent"]

        if primary_intent == QueryIntent.COMPOUND_CHANGE:
            steps.append(
                ExecutionTraceStep(
                    step_id="step_temporal_align",
                    step_name="Multitemporal Geometric & Radiometric Alignment",
                    tool_name="temporal_aligner",
                    status="pending",
                    summary="Normalizing solar elevation, resampling grids, and aligning histograms",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_cva",
                    step_name="Change Vector Analysis (CVA) Magnitude",
                    tool_name="cva_detector",
                    status="pending",
                    summary="Quantifying Euclidean spectral shift and calculating Otsu separability (η)",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_ndvi_delta",
                    step_name="Biophysical Vegetation Loss Quantification (NDVI Delta)",
                    tool_name="ndvi_analyzer",
                    status="pending",
                    summary="Tracking chlorophyll absorption drop and vegetative canopy reduction",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_builtup_gain",
                    step_name="Built-up Expansion & Radiometric Brightness Shift",
                    tool_name="builtup_analyzer",
                    status="pending",
                    summary="Isolating spectral brightening and structural boundary formation",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_spatial_intersection",
                    step_name="Spatial Intersection & Hotspot Grounding",
                    tool_name="spatial_intersection",
                    status="pending",
                    summary="Intersecting vegetation loss with new construction to delineate converted tracts",
                )
            )

        elif is_multitemporal or primary_intent == QueryIntent.CHANGE_DETECTION:
            steps.append(
                ExecutionTraceStep(
                    step_id="step_temporal_align",
                    step_name="Multitemporal Geometric & Radiometric Alignment",
                    tool_name="temporal_aligner",
                    status="pending",
                    summary="Normalizing solar elevation and coregistering pixel grids",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_cva",
                    step_name="Change Vector Analysis (CVA) & Delta Classification",
                    tool_name="cva_detector",
                    status="pending",
                    summary="Quantifying Euclidean spectral shift and calculating Otsu separability",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_change_grounding",
                    step_name="Spatial Localization of Changed Sectors",
                    tool_name="spatial_grounder",
                    status="pending",
                    summary="Extracting change clusters, polygon boundaries, and physical area estimates",
                )
            )

        elif is_optical_sar or primary_intent == QueryIntent.OPTICAL_SAR:
            steps.append(
                ExecutionTraceStep(
                    step_id="step_sar_speckle",
                    step_name="SAR Speckle Filtering & Decibel Calibration",
                    tool_name="sar_analyzer",
                    status="pending",
                    summary="Applying Lee adaptive filter and converting backscatter to sigma nought (dB)",
                )
            )
            steps.append(
                ExecutionTraceStep(
                    step_id="step_cross_modal_fusion",
                    step_name="Optical-SAR Dielectric & Spectral Correlation",
                    tool_name="multimodal_fusion",
                    status="pending",
                    summary="Correlating optical structure with microwave double-bounce reflections",
                )
            )

        else:
            # Single-image workflows
            if parsed_query["requires_spectral"] or "vegetation" in parsed_query.get("target_features", []):
                steps.append(
                    ExecutionTraceStep(
                        step_id="step_ndvi",
                        step_name="Normalized Difference Vegetation Index (NDVI) Pipeline",
                        tool_name="ndvi_analyzer",
                        status="pending",
                        summary="Computing vegetation chlorophyll absorption metrics and false-color heatmap",
                    )
                )

            if "water" in parsed_query.get("target_features", []):
                steps.append(
                    ExecutionTraceStep(
                        step_id="step_ndwi",
                        step_name="Normalized Difference Water Index (NDWI) Pipeline",
                        tool_name="water_detector",
                        status="pending",
                        summary="Delineating open water boundaries and spectral absorption contrast",
                    )
                )

            if parsed_query["requires_grounding"]:
                steps.append(
                    ExecutionTraceStep(
                        step_id="step_spatial_ground",
                        step_name="Spatial Grounding & Morphological Segmentation",
                        tool_name="spatial_grounder",
                        status="pending",
                        summary="Tracing connected component contours and generating georeferenced polygon boundaries",
                    )
                )

        # Final Synthesis Step
        steps.append(
            ExecutionTraceStep(
                step_id="step_vlm_synthesis",
                step_name="Evidence-Conditioned Vision-Language Reasoning",
                tool_name="vqa_reasoner",
                status="pending",
                summary="Formulating precise, non-hallucinatory findings grounded strictly in sensor evidence",
            )
        )

        return steps
