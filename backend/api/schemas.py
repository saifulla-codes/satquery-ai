"""Pydantic schemas for request and response payloads across SatQuery AI.

Enforces evidence-first responses, legitimate confidence handling (null when not mathematically derived),
and full geospatial polygon, pixel, and geographic coordinate structures.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class PolygonPoint(BaseModel):
    x: float = Field(description="Normalized X coordinate [0.0 - 1.0]")
    y: float = Field(description="Normalized Y coordinate [0.0 - 1.0]")


class SpatialFeature(BaseModel):
    id: str = Field(default="", description="Unique identifier for the detected feature")
    label: str = Field(description="Class or entity label, e.g. Water Body, Vegetated Tract, Built-up Structure")
    ymin: float = Field(description="Normalized top coordinate [0.0 - 1.0]")
    xmin: float = Field(description="Normalized left coordinate [0.0 - 1.0]")
    ymax: float = Field(description="Normalized bottom coordinate [0.0 - 1.0]")
    xmax: float = Field(description="Normalized right coordinate [0.0 - 1.0]")
    polygon: Optional[List[PolygonPoint]] = Field(default=None, description="Contour polygon points if extracted")
    confidence: Optional[float] = Field(default=None, description="Legitimate confidence score [0.0 - 1.0], or null")
    area_hectares: Optional[float] = Field(default=None, description="Physical area in hectares if georeferenced")
    pixel_bounds: Optional[Dict[str, int]] = Field(default=None, description="Exact pixel coordinates {ymin_px, xmin_px, ymax_px, xmax_px}")
    geo_bounds: Optional[Dict[str, float]] = Field(default=None, description="Geographic bounding coordinates {west, south, east, north} if georeferenced")
    properties: Dict[str, Any] = Field(default_factory=dict)


# Backwards compatibility alias
BoundingBox = SpatialFeature


class GeoMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    channels: int
    band_names: List[str] = Field(default_factory=list)
    has_nir: bool = False
    modality: str = "OPTICAL"  # OPTICAL, MULTISPECTRAL, SAR, UNKNOWN
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None  # {west, south, east, north}
    pixel_size_meters: Optional[float] = None
    data_type: str = "uint8"
    nodata_value: Optional[float] = None
    sensor_name: Optional[str] = None
    acquisition_date: Optional[str] = None
    is_synthetic_fixture: bool = False


class VisualEvidence(BaseModel):
    id: str
    layer_type: str  # mask, heatmap, ndvi, ndwi, change_map, sar_db, overlay
    title: str
    description: str
    layer_url: str
    default_opacity: float = 0.7
    legend: Optional[Dict[str, str]] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTraceStep(BaseModel):
    step_id: str
    step_name: str
    tool_name: str
    status: str = "completed"  # pending, running, completed, skipped, failed
    duration_ms: float = 0.0
    inputs_summary: Optional[Dict[str, Any]] = None
    summary: str = ""
    evidence_generated: Optional[str] = None


# Backwards compatibility alias
PipelineStep = ExecutionTraceStep


class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    url: str
    metadata: GeoMetadata
    message: str = "Image ingested and parsed successfully"


class AnalysisRequest(BaseModel):
    image_id: str
    query: str
    task_hint: Optional[str] = None
    band_mapping: Optional[Dict[str, int]] = None  # e.g. {"red": 0, "green": 1, "blue": 2, "nir": 3}


class AnalysisResponse(BaseModel):
    analysis_id: str
    query: str
    answer: str
    task_type: str
    confidence: Optional[float] = Field(default=None, description="Computed confidence or null if not mathematically derived")
    confidence_method: Optional[str] = Field(default=None, description="Mathematical derivation methodology of confidence score")
    provider: str = Field(default="local_deterministic_geospatial_engine", description="Active AI or domain engine provider")
    analysis_mode: str = Field(default="LOCAL ANALYSIS", description="LOCAL ANALYSIS or REMOTE AI ANALYSIS")
    tools_used: List[str] = Field(default_factory=list)
    visual_evidence: List[VisualEvidence] = Field(default_factory=list)
    evidence: List[VisualEvidence] = Field(default_factory=list)
    bounding_boxes: List[SpatialFeature] = Field(default_factory=list)
    pipeline_steps: List[ExecutionTraceStep] = Field(default_factory=list)
    execution_trace: List[ExecutionTraceStep] = Field(default_factory=list)
    metadata: GeoMetadata
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: Optional[str] = None
    execution_time_ms: float = 0.0

    @model_validator(mode="after")
    def populate_evidence_aliases(self) -> "AnalysisResponse":
        if not self.evidence and self.visual_evidence:
            self.evidence = self.visual_evidence
        if not self.execution_trace and self.pipeline_steps:
            self.execution_trace = self.pipeline_steps
        return self


class ChangeDetectionRequest(BaseModel):
    image_before_id: str
    image_after_id: str
    query: Optional[str] = "What changed between these two images?"


class ChangeDetectionResponse(BaseModel):
    analysis_id: str
    query: str
    answer: str
    change_percentage: float
    significant_change_area_ha: Optional[float] = None
    change_categories: Dict[str, float] = Field(default_factory=dict)
    threshold_quality: Optional[float] = Field(default=None, description="Otsu separability criterion eta [0.0 - 1.0]")
    confidence: Optional[float] = Field(default=None, description="Real computed threshold confidence or null")
    confidence_method: Optional[str] = Field(default=None, description="Mathematical derivation methodology of confidence score")
    provider: str = Field(default="local_deterministic_geospatial_engine")
    analysis_mode: str = Field(default="LOCAL ANALYSIS")
    tools_used: List[str] = Field(default_factory=list)
    visual_evidence: List[VisualEvidence] = Field(default_factory=list)
    evidence: List[VisualEvidence] = Field(default_factory=list)
    bounding_boxes: List[SpatialFeature] = Field(default_factory=list)
    pipeline_steps: List[ExecutionTraceStep] = Field(default_factory=list)
    execution_trace: List[ExecutionTraceStep] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: Optional[str] = None
    execution_time_ms: float = 0.0

    @model_validator(mode="after")
    def populate_evidence_aliases(self) -> "ChangeDetectionResponse":
        if not self.evidence and self.visual_evidence:
            self.evidence = self.visual_evidence
        if not self.execution_trace and self.pipeline_steps:
            self.execution_trace = self.pipeline_steps
        return self


class OpticalSarRequest(BaseModel):
    optical_image_id: str
    sar_image_id: str
    query: Optional[str] = "Compare optical and SAR evidence to identify development."


class OpticalSarResponse(BaseModel):
    analysis_id: str
    query: str
    answer: str
    optical_evidence_summary: str
    sar_evidence_summary: str
    cross_modal_insights: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None)
    confidence_method: Optional[str] = Field(default=None)
    provider: str = Field(default="local_deterministic_geospatial_engine")
    analysis_mode: str = Field(default="LOCAL ANALYSIS")
    tools_used: List[str] = Field(default_factory=list)
    visual_evidence: List[VisualEvidence] = Field(default_factory=list)
    evidence: List[VisualEvidence] = Field(default_factory=list)
    bounding_boxes: List[SpatialFeature] = Field(default_factory=list)
    pipeline_steps: List[ExecutionTraceStep] = Field(default_factory=list)
    execution_trace: List[ExecutionTraceStep] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: Optional[str] = None
    execution_time_ms: float = 0.0

    @model_validator(mode="after")
    def populate_evidence_aliases(self) -> "OpticalSarResponse":
        if not self.evidence and self.visual_evidence:
            self.evidence = self.visual_evidence
        if not self.execution_trace and self.pipeline_steps:
            self.execution_trace = self.pipeline_steps
        return self


class AnalysisRecord(BaseModel):
    id: str
    created_at: str
    type: str  # single, change, sar
    query: str
    summary: str
    tools_used: List[str]
    has_layers: bool
    confidence: Optional[float] = None
    confidence_method: Optional[str] = None
    analysis_mode: str = "LOCAL ANALYSIS"
