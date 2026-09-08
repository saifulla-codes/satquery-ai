import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ImageViewer } from './components/ImageViewer';
import { AnalysisPanel } from './components/AnalysisPanel';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { EvidenceDeck } from './components/EvidenceDeck';
import type {
  SampleDataset,
  AnalysisResponse,
  ChangeDetectionResponse,
  OpticalSarResponse,
  VisualEvidence,
  BoundingBox,
  GeoMetadata,
  PipelineStep,
} from './types';
import {
  fetchSamples,
  analyzeQuery,
  analyzeChangeDetection,
  analyzeOpticalSarPair,
  uploadSatelliteImage,
} from './services/api';

export const App: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'single' | 'change' | 'sar'>('single');
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleDataset | null>(null);

  // Imagery URLs
  const [primaryImageUrl, setPrimaryImageUrl] = useState<string>('');
  const [secondaryImageUrl, setSecondaryImageUrl] = useState<string | undefined>(undefined);

  // Query & Analysis State
  const [query, setQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [toolsUsed, setToolsUsed] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [limitations, setLimitations] = useState<string | null>(null);
  const [executionTimeMs, setExecutionTimeMs] = useState<number | null>(null);

  // Evidence & Overlays
  const [layers, setLayers] = useState<VisualEvidence[]>([]);
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [metadata, setMetadata] = useState<GeoMetadata | undefined>(undefined);
  const [hoveredBoxId, setHoveredBoxId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load samples on mount
  useEffect(() => {
    loadSamples();
  }, []);

  const loadSamples = async () => {
    try {
      const data = await fetchSamples();
      setSamples(data);
      if (data.length > 0) {
        handleSelectSample(data[0]);
      }
    } catch (err) {
      console.error('Failed to load samples', err);
    }
  };

  const handleSelectSample = (sample: SampleDataset) => {
    setSelectedSample(sample);
    setPrimaryImageUrl(sample.image_url);
    setLayers([]);
    setBoundingBoxes([]);
    setPipelineSteps([]);
    setMetrics({});
    setAnswer(null);
    setLimitations(null);

    // Contextual mode switching
    if (sample.paired_temporal_id) {
      setActiveMode('change');
      const paired = samples.find((s) => s.id === sample.paired_temporal_id);
      setSecondaryImageUrl(paired ? paired.image_url : undefined);
    } else if (sample.paired_sar_id || sample.paired_optical_id) {
      setActiveMode('sar');
      const pairedId = sample.paired_sar_id || sample.paired_optical_id;
      const paired = samples.find((s) => s.id === pairedId);
      setSecondaryImageUrl(paired ? paired.image_url : undefined);
    } else {
      setActiveMode('single');
      setSecondaryImageUrl(undefined);
    }

    // Set first preset query
    if (sample.preset_queries.length > 0) {
      setQuery(sample.preset_queries[0]);
    }
  };

  const handleModeChange = (mode: 'single' | 'change' | 'sar') => {
    setActiveMode(mode);
    setLayers([]);
    setBoundingBoxes([]);
    setPipelineSteps([]);
    setAnswer(null);

    if (mode === 'change') {
      const t1 = samples.find((s) => s.id === 'sample_urban_2021');
      const t2 = samples.find((s) => s.id === 'sample_urban_2025');
      if (t1 && t2) {
        setSelectedSample(t1);
        setPrimaryImageUrl(t1.image_url);
        setSecondaryImageUrl(t2.image_url);
        setQuery('What changed between these two images?');
      }
    } else if (mode === 'sar') {
      const opt = samples.find((s) => s.id === 'sample_harbor_optical');
      const sar = samples.find((s) => s.id === 'sample_harbor_sar');
      if (opt && sar) {
        setSelectedSample(opt);
        setPrimaryImageUrl(opt.image_url);
        setSecondaryImageUrl(sar.image_url);
        setQuery('Compare optical and SAR evidence to identify development.');
      }
    } else {
      const single = samples.find((s) => s.id === 'sample_coastal_multispectral') || samples[0];
      if (single) {
        setSelectedSample(single);
        setPrimaryImageUrl(single.image_url);
        setSecondaryImageUrl(undefined);
        setQuery(single.preset_queries[0] || 'Describe this region.');
      }
    }
  };

  // Quick Demo Trigger
  const handleSelectDemo = async (demoId: number) => {
    if (demoId === 1) {
      // Demo 1: Captioning / Overview
      handleModeChange('single');
      const s = samples.find((x) => x.id === 'sample_coastal_multispectral') || samples[0];
      handleSelectSample(s);
      setQuery('Describe this region and classify visible land cover.');
      setTimeout(() => executeQuery('Describe this region and classify visible land cover.'), 100);
    } else if (demoId === 2) {
      // Demo 2: Water Grounding
      handleModeChange('single');
      const s = samples.find((x) => x.id === 'sample_coastal_multispectral') || samples[0];
      handleSelectSample(s);
      setQuery('Where are the water bodies located?');
      setTimeout(() => executeQuery('Where are the water bodies located?'), 100);
    } else if (demoId === 3) {
      // Demo 3: NDVI Spectral
      handleModeChange('single');
      const s = samples.find((x) => x.id === 'sample_coastal_multispectral') || samples[0];
      handleSelectSample(s);
      setQuery('Where is vegetation concentrated? Calculate NDVI.');
      setTimeout(() => executeQuery('Where is vegetation concentrated? Calculate NDVI.'), 100);
    } else if (demoId === 4) {
      // Demo 4: Multitemporal Change
      handleModeChange('change');
      setQuery('What changed between these two images?');
      setTimeout(() => executeChangeDetection('What changed between these two images?'), 100);
    } else if (demoId === 5) {
      // Demo 5: Optical + SAR
      handleModeChange('sar');
      setQuery('Compare optical and SAR evidence to identify development.');
      setTimeout(() => executeOpticalSar('Compare optical and SAR evidence to identify development.'), 100);
    } else if (demoId === 6) {
      // Demo 6: KILLER DEMO - Complex Multi-Step Agentic Analysis
      handleModeChange('change');
      const killerQuery =
        'Identify areas where vegetation decreased while built-up development increased between the two dates, and show me the affected regions.';
      setQuery(killerQuery);
      setTimeout(() => executeChangeDetection(killerQuery), 100);
    }
  };

  // Execution Handlers
  const handleRunAnalysis = () => {
    if (!query.trim()) return;
    if (activeMode === 'change') {
      executeChangeDetection(query);
    } else if (activeMode === 'sar') {
      executeOpticalSar(query);
    } else {
      executeQuery(query);
    }
  };

  const executeQuery = async (q: string) => {
    if (!selectedSample) return;
    setIsLoading(true);
    setAnswer(null);
    try {
      const resp: AnalysisResponse = await analyzeQuery(selectedSample.id, q);
      setAnswer(resp.answer);
      setToolsUsed(resp.tools_used);
      setConfidence(resp.confidence ?? null);
      setLimitations(resp.limitations ?? null);
      setExecutionTimeMs(resp.execution_time_ms);
      setLayers(resp.visual_evidence);
      setBoundingBoxes(resp.bounding_boxes);
      setPipelineSteps(resp.pipeline_steps);
      setMetadata(resp.metadata);

      // Extract metrics from visual layers
      const aggregated: Record<string, any> = {};
      resp.visual_evidence.forEach((l) => Object.assign(aggregated, l.statistics));
      setMetrics(aggregated);
    } catch (err: any) {
      setAnswer(`### Execution Error\n${err.message || 'Analysis failed'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const executeChangeDetection = async (q: string) => {
    setIsLoading(true);
    setAnswer(null);
    try {
      const resp: ChangeDetectionResponse = await analyzeChangeDetection(
        'sample_urban_2021',
        'sample_urban_2025',
        q
      );
      setAnswer(resp.answer);
      const tools = resp.pipeline_steps && resp.pipeline_steps.length > 0
        ? resp.pipeline_steps.map((s) => s.tool_name)
        : ['geotiff_decoder', 'raster_aligner', 'change_detector', 'spatial_grounder', 'vlm_reasoner'];
      setToolsUsed(tools);
      setConfidence(resp.confidence ?? (resp.threshold_quality ? Math.round(resp.threshold_quality * 100) / 100 : null));
      setLimitations(resp.limitations ?? null);
      setExecutionTimeMs(resp.execution_time_ms);
      setLayers(resp.visual_evidence);
      setBoundingBoxes(resp.bounding_boxes);
      setPipelineSteps(resp.pipeline_steps);
      setMetrics({
        'Total Change': `${resp.change_percentage}%`,
        'Separability (η)': resp.threshold_quality ? `${Math.round(resp.threshold_quality * 1000) / 10}%` : 'N/A',
        'Area Altered': `${resp.significant_change_area_ha || 0} ha`,
        'New Built-up': `+${resp.change_categories?.builtup_expansion_pct || 0}%`,
        'Vegetation Loss': `-${resp.change_categories?.vegetation_loss_pct || 0}%`,
      });
    } catch (err: any) {
      setAnswer(`### Change Detection Error\n${err.message || 'Failed'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const executeOpticalSar = async (q: string) => {
    setIsLoading(true);
    setAnswer(null);
    try {
      const resp: OpticalSarResponse = await analyzeOpticalSarPair(
        'sample_harbor_optical',
        'sample_harbor_sar',
        q
      );
      setAnswer(resp.answer);
      const tools = resp.pipeline_steps && resp.pipeline_steps.length > 0
        ? resp.pipeline_steps.map((s) => s.tool_name)
        : ['geotiff_decoder', 'sar_speckle_filter', 'optical_sar_fuser', 'spatial_grounder', 'vlm_reasoner'];
      setToolsUsed(tools);
      setConfidence(resp.confidence ?? null);
      setLimitations(resp.limitations ?? null);
      setExecutionTimeMs(resp.execution_time_ms);
      setLayers(resp.visual_evidence);
      setBoundingBoxes(resp.bounding_boxes);
      setPipelineSteps(resp.pipeline_steps);
      setMetrics(
        resp.metrics && Object.keys(resp.metrics).length > 0
          ? resp.metrics
          : {
              'Optical Structures': `${resp.bounding_boxes.length} clusters`,
              'Radar Double-Bounce': '>-6 dB',
              'Speckle Filter': 'Lee Adaptive 5x5',
            }
      );
    } catch (err: any) {
      setAnswer(`### Optical-SAR Fusion Error\n${err.message || 'Failed'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    try {
      const uploadResp = await uploadSatelliteImage(file);
      const newDataset: SampleDataset = {
        id: uploadResp.image_id,
        filename: uploadResp.filename,
        title: `Uploaded: ${uploadResp.filename}`,
        modality: uploadResp.metadata.modality || 'OPTICAL',
        bands: uploadResp.metadata.band_names || ['Band 1'],
        resolution: uploadResp.metadata.pixel_size_meters
          ? `${uploadResp.metadata.pixel_size_meters}m`
          : 'Standard',
        crs: uploadResp.metadata.crs,
        description: `Uploaded satellite raster (${uploadResp.metadata.width}x${uploadResp.metadata.height} px)`,
        preset_queries: [
          'Describe this region and classify visible land cover.',
          'Where are the water bodies located?',
          'Where is vegetation concentrated?',
        ],
        image_url: uploadResp.url,
      };

      setSamples((prev) => [newDataset, ...prev]);
      handleSelectSample(newDataset);
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportReport = () => {
    const reportData = {
      title: 'SatQuery AI — Remote Sensing Intelligence Dossier',
      competition: 'Smart India Hackathon 2026 (SIH26167)',
      organization: 'ISRO Space Technology',
      timestamp: new Date().toISOString(),
      active_mode: activeMode,
      query: query,
      answer: answer,
      confidence: confidence,
      tools_used: toolsUsed,
      limitations: limitations,
      metrics: metrics,
      bounding_boxes: boundingBoxes,
      metadata: metadata,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Intelligence_Report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Hidden File Upload Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileUpload}
        accept=".tif,.tiff,.png,.jpg,.jpeg"
        className="hidden"
      />

      {/* Top Header */}
      <Header onSelectDemo={handleSelectDemo} activeMode={activeMode} />

      {/* Main Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          activeMode={activeMode}
          onModeChange={handleModeChange}
          samples={samples}
          selectedSampleId={selectedSample?.id || ''}
          onSelectSample={handleSelectSample}
          onUploadClick={() => fileInputRef.current?.click()}
        />

        {/* Center Satellite Canvas Viewer + Bottom Panes */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 flex overflow-hidden">
            <ImageViewer
              primaryImageUrl={primaryImageUrl}
              secondaryImageUrl={secondaryImageUrl}
              isSplitMode={activeMode === 'change' || activeMode === 'sar'}
              layers={layers}
              boundingBoxes={boundingBoxes}
              metadata={metadata}
              hoveredBoxId={hoveredBoxId}
              onHoverBox={setHoveredBoxId}
            />

            {/* Right Analysis Panel */}
            <AnalysisPanel
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleRunAnalysis}
              isLoading={isLoading}
              presetQueries={selectedSample?.preset_queries || []}
              answer={answer}
              toolsUsed={toolsUsed}
              confidence={confidence}
              limitations={limitations}
              executionTimeMs={executionTimeMs}
            />
          </div>

          {/* Bottom Diagnostics / Pipeline Visualizer & Evidence Deck */}
          {(pipelineSteps.length > 0 || Object.keys(metrics).length > 0) && (
            <div className="border-t border-slate-800 bg-slate-900/90 backdrop-blur-md p-4 max-h-64 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4">
              <PipelineVisualizer steps={pipelineSteps} />
              <EvidenceDeck
                boundingBoxes={boundingBoxes}
                evidenceLayers={layers}
                metrics={metrics}
                hoveredBoxId={hoveredBoxId}
                onHoverBox={setHoveredBoxId}
                onExportReport={handleExportReport}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
