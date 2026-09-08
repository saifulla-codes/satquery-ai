"""Spectral analysis engine: Calibrated NDVI, NDWI, RGB fallbacks, and procedural colormapping."""

import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from ..raster.geotiff_reader import GeoRaster


class SpectralAnalyzer:
    """Computes calibrated spectral remote sensing indices and visualization overlays."""

    @classmethod
    def calculate_ndvi(cls, raster: GeoRaster) -> Tuple[np.ndarray, Dict[str, Any], Optional[str]]:
        """Calculates Normalized Difference Vegetation Index (NDVI).

        Returns:
            (ndvi_array, metrics_dict, limitations_str)
        """
        red = raster.get_red()
        nir = raster.get_nir()

        res_m = raster.metadata.get("pixel_size_meters") or 10.0

        if nir is not None:
            # Calibrated biophysical multispectral NDVI
            numerator = nir - red
            denominator = nir + red + 1e-6
            ndvi = numerator / denominator

            # Mask invalid / inf / nan values safely
            valid_mask = np.isfinite(ndvi) & (nir >= 0.0) & (red >= 0.0)
            ndvi = np.where(valid_mask, np.clip(ndvi, -1.0, 1.0), 0.0)

            valid_pixels = np.sum(valid_mask)
            total_pixels = max(ndvi.size, 1)

            if valid_pixels > 0:
                ndvi_valid = ndvi[valid_mask]
                mean_v = float(np.mean(ndvi_valid))
                median_v = float(np.median(ndvi_valid))
                min_v = float(np.min(ndvi_valid))
                max_v = float(np.max(ndvi_valid))
                std_v = float(np.std(ndvi_valid))

                dense_mask = (ndvi > 0.50) & valid_mask
                mod_mask = (ndvi >= 0.20) & (ndvi <= 0.50) & valid_mask
                sparse_mask = (ndvi >= 0.05) & (ndvi < 0.20) & valid_mask
                non_veg_mask = (ndvi < 0.05) & valid_mask

                dense_cnt = int(np.sum(dense_mask))
                mod_cnt = int(np.sum(mod_mask))
                sparse_cnt = int(np.sum(sparse_mask))
                veg_total_cnt = dense_cnt + mod_cnt

                total_veg_pct = round(veg_total_cnt / valid_pixels * 100, 2)
                dense_pct = round(dense_cnt / valid_pixels * 100, 2)
                mod_pct = round(mod_cnt / valid_pixels * 100, 2)
                sparse_pct = round(sparse_cnt / valid_pixels * 100, 2)
                non_veg_pct = round(np.sum(non_veg_mask) / valid_pixels * 100, 2)

                veg_ha = round((veg_total_cnt * (res_m ** 2)) / 10000.0, 2)
            else:
                mean_v = median_v = min_v = max_v = std_v = 0.0
                total_veg_pct = dense_pct = mod_pct = sparse_pct = non_veg_pct = 0.0
                veg_ha = 0.0

            metrics = {
                "method": "Calibrated NIR-Red Biophysical NDVI",
                "spectral_bands": "Band NIR vs Band Red",
                "mean_ndvi": round(mean_v, 4),
                "median_ndvi": round(median_v, 4),
                "min_ndvi": round(min_v, 4),
                "max_ndvi": round(max_v, 4),
                "std_ndvi": round(std_v, 4),
                "total_vegetation_pct": total_veg_pct,
                "dense_vegetation_pct": dense_pct,
                "moderate_vegetation_pct": mod_pct,
                "sparse_vegetation_pct": sparse_pct,
                "non_vegetated_pct": non_veg_pct,
                "vegetation_area_hectares": veg_ha,
            }
            return ndvi, metrics, None
        else:
            # Fallback: Visible Atmospherically Resistant Index (VARI) = (Green - Red) / (Green + Red - Blue)
            green = raster.get_green()
            blue = raster.get_blue()

            denom = green + red - blue
            denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
            vari = (green - red) / denom
            valid_mask = np.isfinite(vari)
            vari = np.where(valid_mask, np.clip(vari, -1.0, 1.0), 0.0)

            valid_pixels = max(np.sum(valid_mask), 1)
            vari_valid = vari[valid_mask]
            veg_proxy = (vari > 0.15) & valid_mask
            veg_cnt = int(np.sum(veg_proxy))

            mean_v = float(np.mean(vari_valid))
            est_pct = round(veg_cnt / valid_pixels * 100, 2)
            est_ha = round((veg_cnt * (res_m ** 2)) / 10000.0, 2)

            metrics = {
                "method": "RGB Visible Atmospherically Resistant Index (VARI Proxy)",
                "spectral_bands": "Visible Green / Red / Blue",
                "mean_index": round(mean_v, 4),
                "median_index": round(float(np.median(vari_valid)), 4),
                "min_index": round(float(np.min(vari_valid)), 4),
                "max_index": round(float(np.max(vari_valid)), 4),
                "estimated_vegetation_pct": est_pct,
                "estimated_vegetation_hectares": est_ha,
                "total_vegetation_pct": est_pct,
            }
            limitations = (
                "Near-Infrared (NIR) band is absent in this RGB image. Analysis fell back to "
                "Visible Atmospherically Resistant Index (VARI) greenness proxy. Quantitative cellular "
                "chlorophyll biomass quantification requires true Near-Infrared sensors (e.g. Sentinel-2 Band 8)."
            )
            return vari, metrics, limitations

    @classmethod
    def calculate_ndwi(cls, raster: GeoRaster) -> Tuple[np.ndarray, Dict[str, Any], Optional[str]]:
        """Calculates Normalized Difference Water Index (NDWI).

        Returns:
            (ndwi_array, metrics_dict, limitations_str)
        """
        green = raster.get_green()
        nir = raster.get_nir()
        res_m = raster.metadata.get("pixel_size_meters") or 10.0

        if nir is not None:
            # McFeeters Green-NIR NDWI = (Green - NIR) / (Green + NIR)
            numerator = green - nir
            denominator = green + nir + 1e-6
            ndwi = numerator / denominator
            valid_mask = np.isfinite(ndwi)
            ndwi = np.where(valid_mask, np.clip(ndwi, -1.0, 1.0), 0.0)

            water_mask = (ndwi > 0.05) & valid_mask
            total_pixels = max(ndwi.size, 1)
            water_pixels = int(np.sum(water_mask))
            water_pct = round(water_pixels / total_pixels * 100, 2)
            water_ha = round((water_pixels * (res_m ** 2)) / 10000.0, 2)

            mean_val = float(np.mean(ndwi[water_mask])) if water_pixels > 0 else 0.0

            metrics = {
                "method": "McFeeters Green-NIR NDWI",
                "spectral_bands": "Band Green vs Band NIR",
                "mean_water_ndwi": round(mean_val, 4),
                "water_coverage_pct": water_pct,
                "water_pixel_count": water_pixels,
                "water_area_hectares": water_ha,
            }
            return ndwi, metrics, None
        else:
            # RGB Water Proxy: Blue dominance over Red with low overall Red reflectance
            blue = raster.get_blue()
            red = raster.get_red()
            rgb_ratio = (green - red) / (green + red + 1e-6)
            water_mask = (rgb_ratio > 0.05) & (blue > red * 1.15) & (red < 0.35)

            water_pixels = int(np.sum(water_mask))
            total_pixels = max(water_mask.size, 1)
            water_pct = round(water_pixels / total_pixels * 100, 2)
            water_ha = round((water_pixels * (res_m ** 2)) / 10000.0, 2)

            metrics = {
                "method": "RGB Chromatic Water Ratio (Proxy)",
                "spectral_bands": "Visible Blue & Green chromatic absorption",
                "water_coverage_pct": water_pct,
                "water_pixel_count": water_pixels,
                "water_area_hectares": water_ha,
            }
            limitations = (
                "NIR band is absent in standard RGB image. Water delineation used spectral chromatic "
                "absorption contrast. Clear deep water vs dark asphalt/shadow may exhibit minor overlap."
            )
            return rgb_ratio, metrics, limitations

    @classmethod
    def render_colormap_rgba(
        cls,
        data: np.ndarray,
        colormap: str = "rdylgn",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        transparent_below: Optional[float] = None,
    ) -> Image.Image:
        """Converts a 2D float index into an RGBA Pillow Image with smooth colormapping."""
        arr = np.nan_to_num(data, nan=0.0)
        c_min = vmin if vmin is not None else float(np.min(arr))
        c_max = vmax if vmax is not None else float(np.max(arr))

        if c_max <= c_min:
            c_max = c_min + 1e-4

        norm = np.clip((arr - c_min) / (c_max - c_min), 0.0, 1.0)
        h, w = norm.shape

        r = np.zeros((h, w), dtype=np.uint8)
        g = np.zeros((h, w), dtype=np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)
        a = np.full((h, w), 255, dtype=np.uint8)

        if colormap == "rdylgn":
            low_mask = norm < 0.5
            t_low = norm[low_mask] * 2.0
            r[low_mask] = (215 + t_low * (254 - 215)).astype(np.uint8)
            g[low_mask] = (48 + t_low * (224 - 48)).astype(np.uint8)
            b[low_mask] = (39 + t_low * (139 - 39)).astype(np.uint8)

            high_mask = ~low_mask
            t_high = (norm[high_mask] - 0.5) * 2.0
            r[high_mask] = (254 - t_high * (254 - 26)).astype(np.uint8)
            g[high_mask] = (224 - t_high * (224 - 152)).astype(np.uint8)
            b[high_mask] = (139 - t_high * (139 - 80)).astype(np.uint8)

        elif colormap == "blues":
            r = (235 - norm * 215).astype(np.uint8)
            g = (245 - norm * 145).astype(np.uint8)
            b = (255 - norm * 25).astype(np.uint8)

        elif colormap == "change":
            diff = norm - 0.5
            neg_mask = diff < -0.05
            pos_mask = diff > 0.05
            neutral_mask = ~(neg_mask | pos_mask)

            # Red for loss
            r[neg_mask] = 239
            g[neg_mask] = 68
            b[neg_mask] = 68
            a[neg_mask] = np.clip(np.abs(diff[neg_mask]) * 2.0 * 255, 120, 240).astype(np.uint8)

            # Green for growth
            r[pos_mask] = 34
            g[pos_mask] = 197
            b[pos_mask] = 94
            a[pos_mask] = np.clip(diff[pos_mask] * 2.0 * 255, 120, 240).astype(np.uint8)

            a[neutral_mask] = 0
            return Image.fromarray(np.stack([r, g, b, a], axis=-1), mode="RGBA")

        elif colormap == "sar_plasma":
            r = (norm * 255).astype(np.uint8)
            g = (np.sin(norm * np.pi) * 220).astype(np.uint8)
            b = ((1.0 - norm) * 230).astype(np.uint8)

        else:
            r = (68 + norm * (253 - 68)).astype(np.uint8)
            g = (1 + norm * (231 - 1)).astype(np.uint8)
            b = (84 + (1.0 - norm) * 180).astype(np.uint8)

        if transparent_below is not None:
            a[arr < transparent_below] = 0

        rgba = np.stack([r, g, b, a], axis=-1)
        return Image.fromarray(rgba, mode="RGBA")
