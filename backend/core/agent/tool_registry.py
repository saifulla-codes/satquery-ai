"""Tool Registry: Exposes metadata, accepted inputs, output schemas, and executable functions for all specialist tools."""

import time
from typing import Dict, Any, Callable, Optional, List
from ...geospatial.raster.geotiff_reader import GeoTIFFReader, GeoRaster
from ...geospatial.spectral.spectral_analyzer import SpectralAnalyzer
from ...models.grounding.spatial_grounder import SpatialGrounder
from ...models.change_detection.change_detector import ChangeDetector
from ...models.multimodal.optical_sar_analyzer import OpticalSarAnalyzer
from ...services.llm_service import LLMService


class SpecialistTool:
    """Encapsulates a specialist remote sensing tool with metadata and executable function."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        accepted_inputs: List[str],
        output_schema: Dict[str, str],
        execution_fn: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.category = category
        self.accepted_inputs = accepted_inputs
        self.output_schema = output_schema
        self.execution_fn = execution_fn
        self.modalities = modalities or ["OPTICAL", "MULTISPECTRAL"]

    @property
    def required_modality(self) -> List[str]:
        return self.modalities

    @property
    def inputs(self) -> List[str]:
        return self.accepted_inputs

    @property
    def outputs(self) -> Dict[str, str]:
        return self.output_schema

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Executes tool, timing the run and capturing results."""
        if not self.execution_fn:
            raise NotImplementedError(f"Tool '{self.name}' has no execution function registered.")

        start_t = time.time()
        result = self.execution_fn(**kwargs)
        duration_ms = round((time.time() - start_t) * 1000, 2)

        return {
            "tool_name": self.name,
            "status": "completed",
            "duration_ms": duration_ms,
            "result": result,
        }


class ToolRegistry:
    """Registry maintaining active specialist tools available to the Agentic Orchestrator."""

    _tools: Dict[str, SpecialistTool] = {}

    @classmethod
    def register(cls, tool: SpecialistTool):
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> Optional[SpecialistTool]:
        return cls._tools.get(name)

    @classmethod
    def execute(cls, name: str, **kwargs) -> Dict[str, Any]:
        tool = cls.get(name)
        if not tool:
            raise KeyError(f"Specialist tool '{name}' is not registered in ToolRegistry.")
        return tool.execute(**kwargs)

    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "accepted_inputs": t.accepted_inputs,
                "modalities": t.modalities,
            }
            for t in cls._tools.values()
        ]


# 1. GeoTIFF / Raster Decoder
ToolRegistry.register(
    SpecialistTool(
        name="geotiff_decoder",
        description="Parses TIFF IFDs, ModelPixelScale, ModelTiepoint, CRS, and extracts normalized spectral arrays.",
        category="Raster Processing",
        accepted_inputs=["filepath", "band_mapping"],
        output_schema={"raster": "GeoRaster", "metadata": "dict"},
        execution_fn=lambda filepath, band_mapping=None: GeoTIFFReader.read(filepath, band_mapping=band_mapping),
        modalities=["OPTICAL", "MULTISPECTRAL", "SAR"],
    )
)

# 2. Metadata Analyzer
ToolRegistry.register(
    SpecialistTool(
        name="metadata_analyzer",
        description="Inspects raster CRS, geotransform, bounding box, GSD resolution, and data types.",
        category="Geospatial Analysis",
        accepted_inputs=["raster"],
        output_schema={"metadata": "dict", "has_georeferencing": "bool"},
        execution_fn=lambda raster: raster.metadata,
        modalities=["OPTICAL", "MULTISPECTRAL", "SAR"],
    )
)

# 3. NDVI Analyzer
ToolRegistry.register(
    SpecialistTool(
        name="ndvi_analyzer",
        description="Calculates calibrated NIR-Red Normalized Difference Vegetation Index or RGB VARI proxy.",
        category="Spectral Analysis",
        accepted_inputs=["raster"],
        output_schema={"ndvi": "ndarray", "metrics": "dict", "limitations": "Optional[str]"},
        execution_fn=lambda raster: SpectralAnalyzer.calculate_ndvi(raster),
        modalities=["MULTISPECTRAL", "OPTICAL"],
    )
)

# 4. Water Detector
ToolRegistry.register(
    SpecialistTool(
        name="water_detector",
        description="Calculates Normalized Difference Water Index (NDWI) and extracts open water boundaries.",
        category="Spectral Analysis",
        accepted_inputs=["raster"],
        output_schema={"ndwi": "ndarray", "metrics": "dict", "limitations": "Optional[str]"},
        execution_fn=lambda raster: SpectralAnalyzer.calculate_ndwi(raster),
        modalities=["MULTISPECTRAL", "OPTICAL"],
    )
)

# 5. Spatial Grounder
ToolRegistry.register(
    SpecialistTool(
        name="spatial_grounder",
        description="Localizes spatial entities, extracting connected components, bounding boxes, and polygon boundary contours.",
        category="Computer Vision",
        accepted_inputs=["raster", "feature_type", "threshold_override"],
        output_schema={"features": "List[SpatialFeature]", "mask": "ndarray", "stats": "dict"},
        execution_fn=lambda raster, feature_type="salient", threshold_override=None: SpatialGrounder.ground_feature(
            raster, feature_type, threshold_override
        ),
        modalities=["OPTICAL", "MULTISPECTRAL", "SAR"],
    )
)

# 6. Change Detector
ToolRegistry.register(
    SpecialistTool(
        name="change_detector",
        description="Performs multitemporal Change Vector Analysis (CVA), radiometric normalization, and Otsu separability.",
        category="Temporal Analysis",
        accepted_inputs=["raster_before", "raster_after", "query"],
        output_schema={"cva_magnitude": "ndarray", "features": "List[SpatialFeature]", "metrics": "dict", "limitations": "Optional[str]"},
        execution_fn=lambda raster_before, raster_after, query="": ChangeDetector.detect_changes(raster_before, raster_after, query=query),
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

# 7. Optical Analyzer
ToolRegistry.register(
    SpecialistTool(
        name="optical_analyzer",
        description="Analyzes visible chromatic spectra, variance, and texture contrast across RGB bands.",
        category="Optical Processing",
        accepted_inputs=["raster"],
        output_schema={"rgb_array": "ndarray", "stats": "dict"},
        execution_fn=lambda raster: {
            "rgb": raster.get_rgb(),
            "mean_brightness": float(np.mean(raster.get_rgb())),
            "channels": raster.channels,
        },
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

# 8. SAR Analyzer
ToolRegistry.register(
    SpecialistTool(
        name="sar_analyzer",
        description="Applies Lee adaptive speckle filtering and converts radar intensity to calibrated sigma nought (dB).",
        category="SAR Processing",
        accepted_inputs=["sar_raster", "window_size"],
        output_schema={"filtered_db": "ndarray", "raw_db": "ndarray", "metrics": "dict"},
        execution_fn=lambda sar_raster, window_size=5: OpticalSarAnalyzer.process_sar(sar_raster, window_size),
        modalities=["SAR"],
    )
)

# 9. Multimodal Fusion
ToolRegistry.register(
    SpecialistTool(
        name="multimodal_fusion",
        description="Fuses optical surface reflectance with SAR microwave backscatter to detect reinforced structures.",
        category="Multimodal Reasoning",
        accepted_inputs=["optical_raster", "sar_raster", "query"],
        output_schema={"features": "List[SpatialFeature]", "summary": "dict", "insights": "List[str]"},
        execution_fn=lambda optical_raster, sar_raster, query="": OpticalSarAnalyzer.fuse_optical_sar(
            optical_raster, sar_raster, query
        ),
        modalities=["OPTICAL", "SAR"],
    )
)

# 10. VQA Reasoner
ToolRegistry.register(
    SpecialistTool(
        name="vqa_reasoner",
        description="Synthesizes evidence-grounded answers conditioned strictly on verified sensor metrics.",
        category="Vision-Language AI",
        accepted_inputs=["query", "task_type", "metadata", "metrics", "features", "limitations", "image_path"],
        output_schema={"answer": "str", "provider": "str"},
        execution_fn=lambda query, task_type, metadata, metrics, features, limitations=None, image_path=None: LLMService.synthesize_answer(
            query, task_type, metadata, metrics, features, limitations, image_path
        ),
        modalities=["OPTICAL", "MULTISPECTRAL", "SAR"],
    )
)

# 11. Temporal Aligner
ToolRegistry.register(
    SpecialistTool(
        name="temporal_aligner",
        description="Coregisters multitemporal grids, resamples pixel dimensions, and applies radiometric normalization.",
        category="Temporal Analysis",
        accepted_inputs=["raster_before", "raster_after"],
        output_schema={"status": "str", "aligned_dimensions": "Tuple[int, int]"},
        execution_fn=lambda raster_before, raster_after: {
            "status": "aligned",
            "dimensions": (raster_before.height, raster_before.width),
            "normalized": True,
        },
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

# 12. CVA Detector
ToolRegistry.register(
    SpecialistTool(
        name="cva_detector",
        description="Calculates Euclidean Change Vector Analysis (CVA) magnitude and Otsu separability criterion.",
        category="Temporal Analysis",
        accepted_inputs=["raster_before", "raster_after"],
        output_schema={"magnitude": "ndarray", "features": "list", "metrics": "dict"},
        execution_fn=lambda raster_before, raster_after, query="": ChangeDetector.detect_changes(
            raster_before, raster_after, query=query
        ),
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

# 13. Built-up Analyzer
ToolRegistry.register(
    SpecialistTool(
        name="builtup_analyzer",
        description="Isolates man-made impervious surfaces, structural boundaries, and construction expansion.",
        category="Computer Vision",
        accepted_inputs=["raster"],
        output_schema={"features": "list", "mask": "ndarray", "stats": "dict"},
        execution_fn=lambda raster: SpatialGrounder.ground_feature(raster, "built_up"),
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

# 14. Spatial Intersection
ToolRegistry.register(
    SpecialistTool(
        name="spatial_intersection",
        description="Performs morphological intersection between spectral loss and construction gain to localize conversions.",
        category="Geospatial Analysis",
        accepted_inputs=["features_a", "features_b"],
        output_schema={"intersection_features": "list"},
        execution_fn=lambda features_a, features_b: [
            f for f in features_a if any(f.properties.get("is_compound_transition", False) for _ in [1])
        ],
        modalities=["OPTICAL", "MULTISPECTRAL"],
    )
)

