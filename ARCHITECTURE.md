# SatQuery AI — System Architecture
**SIH Problem Statement SIH26167 | Indian Space Research Organisation (ISRO)**
*Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries*

---

## 1. Executive Summary & Philosophy
SatQuery AI is designed as a **mission-grade, evidence-grounded satellite intelligence platform**. Unlike generic chatbots that pass raw images directly to large language models (LLMs) with high risk of hallucinating geospatial features, SatQuery AI implements an **agentic hybrid architecture**:

1. **Local High-Performance Geospatial Engine**: Runs on Apple Silicon / local CPU/GPU using NumPy, SciPy, Pillow, and Scikit-learn. Handles raster decoding (GeoTIFF/TIFF/PNG/JPEG), georeferenced coordinate transforms, spectral analysis (NDVI, NDWI, NDBI), change vector analysis (CVA), structural similarity, and morphological spatial grounding.
2. **Specialist Tool Registry & Agent Orchestrator**: Parses natural-language user queries, decomposes them into explicit geospatial analytical subtasks, dispatches the required specialized computational tools, and aggregates hard physical evidence.
3. **Pluggable Vision-Language Reasoning Layer**: Provides provider-agnostic interfaces (Groq, OpenAI, Anthropic, local VLM, RS-VLM) to generate fluent, context-aware remote sensing descriptions *strictly conditioned on verified visual and spectral evidence*.
4. **Professional Space-Tech UI**: High-fidelity dual/single canvas viewer with real-time multi-layer blending (RGB, False Color Infrared, NDVI heatmaps, change masks, spatial bounding boxes), step-by-step pipeline visualization, and downloadable GeoJSON/PDF intelligence reports.

---

## 2. System Architecture Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                    SATQUERY AI USER INTERFACE                                      |
|  [Sidebar Navigation]  |  [Dual-Canvas Raster Viewer]  |  [Agent Chat & Query]  |  [Evidence Deck] |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  │ REST API / SSE
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                   FASTAPI / FLASK REST BACKEND                                     |
|  - /api/upload          - /api/analyze         - /api/change-detection                             |
|  - /api/optical-sar     - /api/samples         - /api/health                                       |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  AGENTIC ORCHESTRATION LAYER                                       |
|                                                                                                    |
|    Natural Query ──► [ Query Understanding & Intent Classifier ]                                   |
|                                     │                                                              |
|                                     ▼                                                              |
|                            [ Task Planner & DAG ]                                                  |
|                                     │                                                              |
|                     ┌───────────────┴───────────────┐                                              |
|                     ▼                               ▼                                              |
|           [ Specialist Tools ]             [ Modality Router ]                                     |
|           - Raster Decoder                 - Optical (RGB / Multi)                                 |
|           - Spectral Indices               - SAR Backscatter                                       |
|           - Change Vector Analysis         - Cross-Modal Pairs                                     |
|           - Spatial Grounding & Masks                                                              |
|                     │                               │                                              |
|                     └───────────────┬───────────────┘                                              |
|                                     ▼                                                              |
|                        [ Evidence Aggregator ]                                                     |
|                  (Masks, BBoxes, Statistics, Coords)                                               |
|                                     │                                                              |
|                                     ▼                                                              |
|                     [ Vision-Language Reasoner ]                                                   |
|                        (Conditioned Synthesis)                                                     |
+----------------------------------------------------------------------------------------------------+
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
+---------------------------------------+   +---------------------------------------+
|        LOCAL GEOSPATIAL ENGINE        |   |         REMOTE / VLM SERVICES         |
|  - GeoTIFF Tag & CRS Extractor        |   |  - Groq Llama-3.2-Vision / Claude     |
|  - Multi-Band Raster Slicer           |   |  - OpenAI GPT-4o / Local Ollama       |
|  - NDVI / NDWI / NDBI Pipeline        |   |  - Remote RS-VLM Worker               |
|  - SciPy Morphological Segmentation   |   |  - Pluggable Fallback Provider        |
|  - CVA & Structural Change Detector   |   |                                       |
|  - SAR Speckle (Lee Filter) & dB Scale|   |                                       |
+---------------------------------------+   +---------------------------------------+
```

---

## 3. Component Breakdown

### A. Geospatial Engine (`backend/geospatial/`)
- `raster/`: GeoTIFF parser decoding IFDs, ModelTiepointTag (33922), ModelPixelScaleTag (33550), GeoKeyDirectoryTag (34735), GDAL metadata, CRS, bounding coordinates, and multi-band arrays.
- `spectral/`: Modality-aware spectral calculators:
  - NDVI: $(NIR - Red) / (NIR + Red)$
  - NDWI: $(Green - NIR) / (Green + NIR)$ (McFeeters) or $(Green - SWIR) / (Green + SWIR)$ (Gao)
  - NDBI: $(SWIR - NIR) / (SWIR + NIR)$
  - Colormap Generator: Converts normalized floating-point rasters into RGBA visualization textures (Viridis, Jet, RdYlGn, Magma) without external graphics dependencies.
- `preprocessing/`: Spatial alignment using normalized cross-correlation, affine transformation, radiometric normalization, and band stacking.

### B. Core Agent & Orchestrator (`backend/core/agent/`)
- `query_parser.py`: Extracts user intents:
  - `SCENE_DESCRIPTION`: Image captioning and contextual land-cover summary.
  - `SPATIAL_GROUNDING`: "Where is...", "Find...", "Detect...", returning bounding boxes & masks.
  - `SPECTRAL_QUERY`: "Vegetation health", "Water bodies", "Urban density".
  - `CHANGE_DETECTION`: Multitemporal comparison between Date $T_1$ and Date $T_2$.
  - `CROSS_MODAL_SAR`: Combined optical surface reflectance + SAR backscatter roughness analysis.
- `planner.py`: Assembles an execution graph of specialist tools.
- `tool_registry.py`: Declarative registry where each specialist module registers its input modalities, required parameters, execution cost, and output evidence schema.

### C. Specialist Models & Algorithms (`backend/models/`)
- `grounding/spatial_grounder.py`: Detects and localizes features (water, vegetation, built-up, vessels/aircraft) via spectral thresholding, connected components (`scipy.ndimage.label`), contour tracing, and polygon approximation.
- `change_detection/change_detector.py`: Computes radiometric difference, spectral index delta ($\Delta\text{NDVI}$), Change Vector Analysis (CVA), and Otsu dynamic thresholding to isolate real physical changes from atmospheric illumination variance.
- `multimodal/optical_sar_analyzer.py`: Filters speckle noise (Lee adaptive filter), converts SAR backscatter to decibel scale $\sigma^0 = 10 \cdot \log_{10}(I)$, and fuses dielectric roughness with optical reflectance.

### D. Vision-Language & Evidence Grounding (`backend/services/`)
- `llm_service.py`: Provider interface supporting Groq, OpenAI, Anthropic, or deterministic offline template synthesizers when running without external network access.
- `evidence_collector.py`: Packages all quantitative metrics (area in hectares, vegetation index distributions, change percentages, geo-coordinates) into a structured JSON payload passed to the UI and reasoner.

### E. Frontend Application (`frontend/`)
- React 19 + TypeScript + Vite + Tailwind CSS.
- Interactive multi-layer canvas viewer supporting real-time pan, zoom, split-screen before/after wiper, opacity sliders, and polygon/bounding-box overlays.
- Dynamic Execution Pipeline widget displaying the agent's real-time step progression.
- Evidence Deck showing metric chips, histogram distributions, and layer export options.
