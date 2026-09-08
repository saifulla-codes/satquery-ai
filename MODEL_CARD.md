# SatQuery AI — Model Card & Algorithmic Specifications
**SIH Problem Statement SIH26167 | ISRO Space Technology**

---

## 1. Algorithmic Overview
SatQuery AI implements a hybrid edge-cloud architecture that avoids the ungrounded hallucinations common to generic LLMs. It combines deterministic physical remote-sensing algorithms with a vision-language orchestrator:

| Model / Algorithm Component | Modality | Processing Paradigm | Hardware Target |
|---|---|---|---|
| **GeoTIFF IFD & CRS Tag Engine** | Multi-Band GeoTIFF / TIFF | Local Pure-Python / NumPy | M2 / CPU (<5ms) |
| **Calibrated NIR/Red NDVI Pipeline** | Multispectral (R, G, B, NIR) | Biophysical Index Slicer | M2 / CPU (<15ms) |
| **McFeeters & Gao NDWI Pipeline** | Multispectral / Optical | Water Delineation Contrast | M2 / CPU (<10ms) |
| **Change Vector Analysis (CVA)** | Multitemporal ($T_1, T_2$) | Euclidean Radiometric Shift | M2 / CPU (<40ms) |
| **Lee Adaptive Speckle Filter** | Synthetic Aperture Radar (SAR) | Spatial Window Covariance | M2 / CPU (<35ms) |
| **Morphological Spatial Grounder** | All Modalities | Connected Component Graphing | M2 / CPU (<20ms) |
| **Vision-Language Orchestrator** | Text + Evidence JSON | Hybrid (Local Reasoner / Remote API) | M2 / Cloud API |

---

## 2. Mathematical Formulations

### A. Normalized Difference Vegetation Index (NDVI)
$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}} + \epsilon}$$
- **Healthy Forest / Crops**: $\text{NDVI} \in [0.45, 0.90]$
- **Sparse Shrub / Grassland**: $\text{NDVI} \in [0.15, 0.45]$
- **Water / Built Structures**: $\text{NDVI} < 0.05$

### B. Visible Atmospherically Resistant Index (VARI — RGB Fallback)
When Near-Infrared (NIR) sensors are unavailable, the system falls back gracefully to VARI:
$$\text{VARI} = \frac{\rho_{\text{Green}} - \rho_{\text{Red}}}{\rho_{\text{Green}} + \rho_{\text{Red}} - \rho_{\text{Blue}} + \epsilon}$$
*Note: A sensor limitation notice is automatically reported to user whenever this fallback is used.*

### C. Change Vector Analysis (CVA)
Multitemporal change magnitude across $N$ co-registered spectral bands:
$$\Delta M = \sqrt{\sum_{b=1}^{N} \left(I_{T_2, b} - I_{T_1, b}\right)^2}$$
Thresholding is calculated dynamically using Otsu's inter-class variance maximization:
$$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$

### D. SAR Decibel Calibration & Lee Filter
Calibrated radar cross section ($\sigma^0$ in dB):
$$\sigma^0\text{ (dB)} = 10 \cdot \log_{10}(I + 10^{-6})$$
Lee adaptive spatial filtering for multiplicative speckle suppression:
$$\hat{x} = \bar{x} + W \cdot (x - \bar{x}), \quad W = \frac{\sigma_x^2}{\sigma_x^2 + \sigma_v^2}$$

---

## 3. Grounding & Confidence Metrics
- **Legitimate Confidence Score**: SatQuery AI computes confidence strictly based on spectral purity, component fill density, and signal-to-noise ratio.
- **Zero Fabrication Policy**: If cloud cover, shadow obscuration, or sensor band omissions prevent definitive classification, the system explicitly reports limitations rather than inventing answers.
