export interface PolygonPoint {
  x: number;
  y: number;
}

export interface BoundingBox {
  id: string;
  label: string;
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  confidence?: number;
  area_hectares?: number;
  polygon?: PolygonPoint[];
  properties?: Record<string, any>;
}

export interface GeoMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
  channels: number;
  band_names: string[];
  has_nir: boolean;
  modality: 'OPTICAL' | 'MULTISPECTRAL' | 'SAR' | 'UNKNOWN';
  crs?: string;
  bounds?: {
    west: number;
    south: number;
    east: number;
    north: number;
  };
  pixel_size_meters?: number;
  sensor_name?: string;
  acquisition_date?: string;
}

export interface VisualEvidence {
  id: string;
  layer_type: 'mask' | 'heatmap' | 'ndvi' | 'ndwi' | 'change_map' | 'sar_db' | 'overlay';
  title: string;
  description: string;
  layer_url: string;
  default_opacity: number;
  legend?: Record<string, string>;
  statistics: Record<string, any>;
}

export interface PipelineStep {
  step_id: string;
  step_name: string;
  tool_name: string;
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'failed';
  duration_ms: number;
  summary: string;
  evidence_generated?: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  query: string;
  answer: string;
  task_type: string;
  confidence?: number;
  provider?: string;
  tools_used: string[];
  visual_evidence: VisualEvidence[];
  bounding_boxes: BoundingBox[];
  pipeline_steps: PipelineStep[];
  metadata: GeoMetadata;
  metrics?: Record<string, any>;
  limitations?: string;
  execution_time_ms: number;
}

export interface ChangeDetectionResponse {
  analysis_id: string;
  query: string;
  answer: string;
  change_percentage: number;
  significant_change_area_ha?: number;
  change_categories: Record<string, number>;
  threshold_quality?: number;
  confidence?: number;
  provider?: string;
  visual_evidence: VisualEvidence[];
  bounding_boxes: BoundingBox[];
  pipeline_steps: PipelineStep[];
  metrics?: Record<string, any>;
  limitations?: string;
  execution_time_ms: number;
}

export interface OpticalSarResponse {
  analysis_id: string;
  query: string;
  answer: string;
  optical_evidence_summary: string;
  sar_evidence_summary: string;
  cross_modal_insights: string[];
  confidence?: number;
  provider?: string;
  visual_evidence: VisualEvidence[];
  bounding_boxes: BoundingBox[];
  pipeline_steps: PipelineStep[];
  metrics?: Record<string, any>;
  limitations?: string;
  execution_time_ms: number;
}

export interface SampleDataset {
  id: string;
  filename: string;
  title: string;
  modality: 'OPTICAL' | 'MULTISPECTRAL' | 'SAR';
  bands: string[];
  resolution: string;
  crs?: string;
  description: string;
  preset_queries: string[];
  image_url: string;
  paired_temporal_id?: string;
  paired_sar_id?: string;
  paired_optical_id?: string;
}
