# SatQuery AI — Remote Sensing Dataset Integration Guide
**SIH Problem Statement SIH26167 | ISRO Space Technology**

---

## 1. Supported Public Benchmark Datasets

SatQuery AI includes dedicated adapter interfaces to integrate standard remote sensing benchmark datasets:

### A. BigEarthNet (Sentinel-2 & Sentinel-1)
- **Modalities**: 12-Band Multispectral optical (Sentinel-2 L2A) and Dual-Polarization SAR (VV, VH from Sentinel-1).
- **Format**: 120x120 pixel patches with 10m, 20m, and 60m ground resolution.
- **Labels**: 43 Corine Land Cover (CLC) classes.
- **Adapter Location**: `datasets/bigearthnet/adapter.py`.

### B. VRSBench
- **Modality**: High-resolution optical satellite & aerial imagery.
- **Annotations**: Visual Question Answering (VQA) pairs, fine-grained remote-sensing captions, and spatial grounding bounding boxes.
- **Adapter Location**: `datasets/vrsbench/adapter.py`.

### C. RSVQA (Remote Sensing Visual Question Answering)
- **Sensors**: Sentinel-2 (Low-resolution 10m) and USGS/NAIP Aerial (High-resolution).
- **Question Types**: Presence, Comparison, Count, Rural vs Urban classification.
- **Adapter Location**: `datasets/rsvqa/adapter.py`.

### D. CDVQA (Change Detection Visual Question Answering)
- **Modality**: Multitemporal optical satellite image pairs ($T_1, T_2$).
- **Annotations**: Grounded questions describing human-induced and environmental land-cover changes.
- **Adapter Location**: `datasets/cdvqa/adapter.py`.

---

## 2. Curated SIH Offline Demonstration Datasets
SatQuery AI includes synthetic, georeferenced satellite datasets bundled locally under `datasets/samples/`:
1. `coastal_port_multispectral.tif`: 4-Band GeoTIFF (Red, Green, Blue, NIR) with real EPSG:4326 geokeys and 10.0m Sentinel-2 GSD.
2. `urban_expansion_2021.png` & `urban_expansion_2025.png`: Multitemporal change detection pair.
3. `industrial_harbor_optical.png` & `industrial_harbor_sar.png`: Co-registered Optical RGB and Sentinel-1 C-Band SAR radar backscatter pair with speckle noise.
