# SatQuery AI — SIH 2026 Presentation & Demonstration Guide
**Problem Statement SIH26167 | ISRO Space Technology**

---

## Executive Demonstration Script for SIH Judges

Open `http://localhost:8000` in any web browser. SatQuery AI features an interactive Quick Demo top bar with 6 curated scenarios:

---

### DEMO 1: Single-Image Visual Captioning & Land-Cover
- **Action**: Click `1. Captioning` in the top bar.
- **Scene Loaded**: `coastal_port_multispectral.tif` (4-band GeoTIFF over coastal harbor).
- **Dispatched Query**: *"Describe this region and classify visible land cover."*
- **What to Observe**:
  - The agent extracts genuine GeoTIFF metadata: 512×512, 4 channels (R, G, B, NIR), CRS `EPSG:4326 (WGS 84)`, 10.0m GSD.
  - The answer categorizes deep water, vegetated hills, and industrial docks.
  - Generates spatial bounding boxes around prominent entities with confidence scores.

---

### DEMO 2: Water Body Spatial Grounding & Delineation
- **Action**: Click `2. Water Grounding`.
- **Dispatched Query**: *"Where are the water bodies located?"*
- **What to Observe**:
  - The agent identifies the target intent as `SPATIAL_GROUNDING` with entity `water`.
  - Computes McFeeters NDWI absorption contrast using NIR and Green bands.
  - Generates a blue water contour mask overlay.
  - Extracts normalized bounding boxes and estimates physical surface area in hectares (~73.6% coverage).

---

### DEMO 3: Multispectral 4-Band NDVI Vegetation Analysis
- **Action**: Click `3. NDVI Spectral`.
- **Dispatched Query**: *"Where is vegetation concentrated? Calculate NDVI."*
- **What to Observe**:
  - Calculates true biophysical Normalized Difference Vegetation Index:
    $$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}$$
  - Generates an interactive RdYlGn (Red-Yellow-Green) false-color heatmap overlay.
  - Reports exact canopy distribution: Dense Vegetation, Moderate Agricultural, and Non-vegetated percentages.
  - Displays opacity slider allowing judges to blend the NDVI layer over the base imagery in real time!

---

### DEMO 4: Multitemporal Change Detection (2021 Baseline vs 2025 Evolution)
- **Action**: Click `4. Change Detection`.
- **Scenes Loaded**: `urban_expansion_2021.png` ($T_1$) and `urban_expansion_2025.png` ($T_2$).
- **Dispatched Query**: *"What changed between these two images?"*
- **What to Observe**:
  - Interactive split-screen wiper slider appears on canvas, allowing judges to drag between 2021 and 2025!
  - Change Vector Analysis (CVA) identifies radiometric shifts.
  - The amber flame change heatmap highlights the newly constructed industrial park in the Northern sector.
  - Bounding boxes isolate the core development zones with area calculations.

---

### DEMO 5: Synthetic Aperture Radar (SAR) & Optical Cross-Modal Reasoning
- **Action**: Click `5. Optical + SAR`.
- **Scenes Loaded**: Optical visible spectrum alongside Sentinel-1 C-Band SAR backscatter.
- **Dispatched Query**: *"Compare optical and SAR evidence to identify development."*
- **What to Observe**:
  - Applies Lee adaptive filter to suppress radar speckle noise.
  - Converts microwave return to decibels ($\sigma^0\text{ dB}$).
  - Corroborates bright double-bounce radar reflections (>-6 dB) with optical structural contrast to confirm reinforced steel/concrete maritime infrastructure.

---

### DEMO 6 — KILLER DEMO: Autonomous Multi-Step Agentic Pipeline
- **Action**: Click `★ 6. Killer Demo`.
- **Dispatched Query**:
  > *"Identify areas where vegetation decreased while built-up development increased between the two dates, and show me the affected regions."*
- **What to Observe**:
  - The **Agentic Orchestration Pipeline** panel visibly tracks each step:
    1. `Query Understanding`: Decomposes multi-part intent into temporal comparison + vegetation delta + built-up gain.
    2. `Raster Engine`: Ingests and aligns temporal rasters.
    3. `Change Vector Analysis`: Computes Euclidean spectral shift.
    4. `Spectral Delta`: Computes $\Delta\text{NDVI}$ and $\Delta\text{Brightness}$.
    5. `Spatial Grounding`: Localizes affected sectors and tags bounding boxes.
    6. `Grounded Synthesis`: Produces structured intelligence report with verifiable metrics.
  - Click **"Export Dossier"** to download the complete JSON intelligence dossier!
