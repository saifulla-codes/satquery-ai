# SatQuery AI — Implementation Plan & Milestone Roadmap
**SIH Problem Statement SIH26167 | ISRO Space Technology**

---

## 1. Milestones and Development Phases

| Phase | Milestone Description | Status | Deliverables |
|---|---|---|---|
| **Phase 1** | **Foundation & Project Infrastructure** | **IN PROGRESS** | Workspace scaffolding, Flask REST API, React-TS Vite frontend, pure-Python GeoTIFF/raster parser, sample satellite imagery generator, `/api/upload` & `/api/health` endpoints |
| **Phase 2** | **VQA & Remote Sensing Captioning Engine** | Scheduled | Scene land-cover descriptor, question parser, prompt engineering with spectral priors, confidence assessment, provider router (Groq/OpenAI/offline fallback) |
| **Phase 3** | **Spatial Grounding & Visual Evidence Generation** | Scheduled | Feature extractors (water bodies, built-up, agriculture, barren), morphological segmentation, bounding boxes, polygon contour vectorization, visual overlay rendering |
| **Phase 4** | **Optical & Multispectral Analysis** | Scheduled | Multi-band TIFF reader (Red, Green, Blue, NIR), band auto-detector, NDVI calculator, NDWI calculator, colormap texture generation (RdYlGn, Viridis) |
| **Phase 5** | **Multitemporal Change Detection** | Scheduled | Before/After image aligner, Change Vector Analysis (CVA), radiometric normalization, Otsu thresholded change masks, natural language change summarizer |
| **Phase 6** | **Optical + SAR Fusion** | Scheduled | SAR backscatter decibel converter ($\sigma^0$), Lee filter speckle reduction, dielectric roughness analysis, joint optical-SAR cross-modal reasoning |
| **Phase 7** | **Agentic Orchestrator & Tool Registry** | Scheduled | Query intent classifier, DAG planner, specialist tool registry, multi-step execution pipeline, evidence aggregator, SIH "Killer Demo" pipeline |
| **Phase 8** | **Mission-Grade Space-Tech UI** | Scheduled | Dark-mode aerospace UI, interactive dual canvas viewer, before/after slider wiper, layer opacity blending, pipeline step progress visualizer, GeoJSON/report exporter |
| **Phase 9** | **Benchmark Dataset Adapters** | Scheduled | Dataset adapter interfaces for BigEarthNet, VRSBench, RSVQA, CDVQA with sample loaders and evaluation metrics |
| **Phase 10** | **SIH Demo Suites & Test Suites** | Scheduled | Deterministic end-to-end demo presets (Urban Growth, Flood/Water Delineation, Deforestation, SAR Co-registration), comprehensive unit & integration tests |

---

## 2. Technical Stack
- **Backend**: Python 3.13, Flask 3.1.3, Pydantic 2.13.5, NumPy 2.5.1, SciPy 1.18.1, Scikit-learn 1.9.0, Pillow 12.2.0, HTTPX 0.28.1
- **Frontend**: Vite 8.2.2, React 19, TypeScript, Tailwind CSS 4
- **Deployment & Runtime**: Apple Silicon M2 (MacBook Air), Local execution with optional cloud VLM routing
