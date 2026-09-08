"""Multitemporal change detection engine: Compatibility check, alignment, Change Vector Analysis (CVA), and Otsu separability."""

import numpy as np
from PIL import Image
from scipy import ndimage
from typing import Dict, Any, Tuple, List, Optional
from ...geospatial.raster.geotiff_reader import GeoRaster
from ...api.schemas import SpatialFeature, PolygonPoint
from ...geospatial.spectral.spectral_analyzer import SpectralAnalyzer


class ChangeDetector:
    """Detects, localizes, and categorizes multitemporal changes between two satellite acquisitions."""

    @classmethod
    def detect_changes(
        cls,
        raster_before: GeoRaster,
        raster_after: GeoRaster,
        query: str = "",
    ) -> Tuple[np.ndarray, List[SpatialFeature], Dict[str, Any], Optional[str]]:
        """Executes full multitemporal change detection pipeline.

        Returns:
            (change_magnitude_arr, spatial_features, metrics_dict, limitations_str)
        """
        # Step 1: Validate compatibility & Align/Resample
        arr_before = raster_before.get_rgb().astype(np.float32) / 255.0
        arr_after = raster_after.get_rgb().astype(np.float32) / 255.0

        h1, w1 = arr_before.shape[:2]
        h2, w2 = arr_after.shape[:2]

        if (h1, w1) != (h2, w2):
            img_after_pil = Image.fromarray((arr_after * 255).astype(np.uint8))
            img_after_resized = img_after_pil.resize((w1, h1), Image.Resampling.BILINEAR)
            arr_after = np.array(img_after_resized).astype(np.float32) / 255.0

        # Step 2: Radiometric normalization (channel-wise mean & variance match)
        for c in range(3):
            mean_b = float(np.mean(arr_before[:, :, c]))
            std_b = float(np.std(arr_before[:, :, c])) + 1e-6
            mean_a = float(np.mean(arr_after[:, :, c]))
            std_a = float(np.std(arr_after[:, :, c])) + 1e-6
            arr_after[:, :, c] = (arr_after[:, :, c] - mean_a) * (std_b / std_a) + mean_b
        arr_after = np.clip(arr_after, 0.0, 1.0)

        # Step 3: Change Vector Analysis (CVA) Euclidean Magnitude across RGB
        diff = arr_after - arr_before
        cva_magnitude = np.sqrt(np.sum(diff ** 2, axis=-1))  # Shape (H, W)

        # Step 4: Spectral direction analysis (NDVI delta if available or VARI proxy)
        ndvi_before, _, _ = SpectralAnalyzer.calculate_ndvi(raster_before)
        ndvi_after, _, _ = SpectralAnalyzer.calculate_ndvi(raster_after)

        if ndvi_before.shape != ndvi_after.shape:
            pil_ndvi_after = Image.fromarray(ndvi_after).resize(
                (ndvi_before.shape[1], ndvi_before.shape[0]), Image.Resampling.BILINEAR
            )
            ndvi_after = np.array(pil_ndvi_after)

        delta_ndvi = ndvi_after - ndvi_before

        brightness_before = np.mean(arr_before, axis=-1)
        brightness_after = np.mean(arr_after, axis=-1)
        delta_brightness = brightness_after - brightness_before

        # Step 5: Dynamic Otsu thresholding with objective separability criterion calculation
        change_threshold, otsu_eta = cls._compute_otsu_threshold(cva_magnitude)

        raw_change_mask = cva_magnitude > change_threshold

        # Morphological filtering to eliminate isolated single-pixel noise
        structure = ndimage.generate_binary_structure(2, 1)
        change_mask = ndimage.binary_opening(raw_change_mask, structure=structure, iterations=1)
        change_mask = ndimage.binary_closing(change_mask, structure=structure, iterations=2)

        # Step 6: Categorize changes
        total_pixels = h1 * w1
        changed_pixels = int(np.sum(change_mask))
        change_pct = round(changed_pixels / max(total_pixels, 1) * 100, 2)

        veg_loss_mask = change_mask & (delta_ndvi < -0.15)
        builtup_gain_mask = change_mask & (delta_brightness > 0.15) & (delta_ndvi <= 0.05)
        water_change_mask = change_mask & (delta_brightness < -0.2) & (~veg_loss_mask)

        # Detect compound transition: Vegetation loss correlated with builtup expansion
        compound_query = any(w in query.lower() for w in ["vegetation", "green"]) and any(
            w in query.lower() for w in ["built", "construct", "develop", "building"]
        )

        veg_loss_pct = round(float(np.sum(veg_loss_mask) / max(total_pixels, 1) * 100), 2)
        builtup_gain_pct = round(float(np.sum(builtup_gain_mask) / max(total_pixels, 1) * 100), 2)
        water_change_pct = round(float(np.sum(water_change_mask) / max(total_pixels, 1) * 100), 2)

        # Step 7: Connected components & polygon boundaries
        labeled_arr, num_features = ndimage.label(change_mask)
        component_sizes = ndimage.sum(change_mask, labeled_arr, range(1, num_features + 1))
        min_cluster = max(int(total_pixels * 0.002), 30)

        sorted_indices = np.argsort(-np.array(component_sizes))[:8]
        slices = ndimage.find_objects(labeled_arr)
        features: List[SpatialFeature] = []

        res_m = raster_before.metadata.get("pixel_size_meters") or 10.0
        bounds = raster_before.metadata.get("bounds")

        compound_transitions_count = 0
        compound_converted_pixels = 0

        for rank, c_idx in enumerate(sorted_indices):
            comp_id = c_idx + 1
            px_count = component_sizes[c_idx]
            if px_count < min_cluster:
                continue

            if c_idx >= len(slices) or slices[c_idx] is None:
                continue

            slice_y, slice_x = slices[c_idx]
            comp_mask = labeled_arr == comp_id

            ymin = float(slice_y.start / h1)
            xmin = float(slice_x.start / w1)
            ymax = float(slice_y.stop / h1)
            xmax = float(slice_x.stop / w1)

            # Categorization within this cluster
            cluster_veg_loss = int(np.sum(comp_mask & veg_loss_mask))
            cluster_builtup = int(np.sum(comp_mask & builtup_gain_mask))

            is_compound = (cluster_builtup > 0 and cluster_veg_loss > 0) or (compound_query and cluster_builtup > 0)
            if is_compound:
                compound_transitions_count += 1
                compound_converted_pixels += px_count

            if compound_query and is_compound:
                category = "Vegetation Converted to Built-up Development"
            elif cluster_builtup > cluster_veg_loss:
                category = "New Built-up Development"
            elif cluster_veg_loss > 0:
                category = "Vegetation Loss / Clearing"
            else:
                category = "Land-cover Modification"

            area_ha = round(float(px_count * (res_m ** 2)) / 10000.0, 2)
            quadrant = cls._get_quadrant((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)

            # Defensible confidence: combines Otsu separability with component fill density
            sub_mask = change_mask[slice_y, slice_x]
            fill_density = float(np.sum(sub_mask) / max(sub_mask.size, 1))
            cluster_conf = round(min(0.60 + otsu_eta * 0.25 + fill_density * 0.15, 0.98), 2)

            # Extract polygon boundary points
            sub_binary = (labeled_arr[slice_y, slice_x] == comp_id)
            polygon_pts = cls._extract_polygon(sub_binary, slice_y.start, slice_x.start, h1, w1)

            pixel_bounds = {
                "ymin_px": int(slice_y.start),
                "xmin_px": int(slice_x.start),
                "ymax_px": int(slice_y.stop),
                "xmax_px": int(slice_x.stop),
            }

            geo_bounds = None
            if bounds and isinstance(bounds, dict) and "west" in bounds:
                w_deg = bounds["east"] - bounds["west"]
                h_deg = bounds["north"] - bounds["south"]
                geo_bounds = {
                    "west": round(bounds["west"] + xmin * w_deg, 6),
                    "east": round(bounds["west"] + xmax * w_deg, 6),
                    "north": round(bounds["north"] - ymin * h_deg, 6),
                    "south": round(bounds["north"] - ymax * h_deg, 6),
                }

            features.append(
                SpatialFeature(
                    id=f"change_{rank+1}",
                    label=f"{category} ({quadrant})",
                    ymin=round(ymin, 4),
                    xmin=round(xmin, 4),
                    ymax=round(ymax, 4),
                    xmax=round(xmax, 4),
                    polygon=polygon_pts,
                    confidence=cluster_conf,
                    area_hectares=area_ha if area_ha > 0.01 else None,
                    pixel_bounds=pixel_bounds,
                    geo_bounds=geo_bounds,
                    properties={
                        "category": category,
                        "quadrant": quadrant,
                        "pixel_count": int(px_count),
                        "otsu_separability": round(otsu_eta, 3),
                        "vegetation_loss_pixels": cluster_veg_loss,
                        "builtup_gain_pixels": cluster_builtup,
                        "is_compound_transition": is_compound,
                        "coordinate_reference": "EPSG:4326" if geo_bounds else "Image Pixel Space",
                    },
                )
            )

        compound_ha = round(float(compound_converted_pixels * (res_m ** 2)) / 10000.0, 2)
        compound_pct = round(compound_converted_pixels / max(total_pixels, 1) * 100, 2)

        metrics = {
            "total_change_pct": change_pct,
            "vegetation_loss_pct": veg_loss_pct,
            "builtup_expansion_pct": builtup_gain_pct,
            "water_fluctuation_pct": water_change_pct,
            "compound_transitions_count": compound_transitions_count,
            "compound_converted_hectares": compound_ha,
            "compound_converted_pct": compound_pct,
            "total_changed_hectares": round(float(changed_pixels * (res_m ** 2)) / 10000.0, 2),
            "clusters_count": len(features),
            "mean_cva_magnitude": round(float(np.mean(cva_magnitude)), 4),
            "threshold_used": round(change_threshold, 4),
            "otsu_separability_eta": round(otsu_eta, 4),
            "confidence_method": f"Dynamic Otsu separability criterion (η={round(otsu_eta, 4)}) with morphological cluster coherence",
        }

        limitations = None
        if not raster_before.has_nir or not raster_after.has_nir:
            limitations = (
                "At least one image lacks a Near-Infrared (NIR) band. Vegetation delta was approximated "
                "via visible spectral channels. Seasonal phenology or agricultural crop cycles may contribute "
                "to detected spectral shifts."
            )

        return cva_magnitude, features, metrics, limitations

    @classmethod
    def _compute_otsu_threshold(cls, cva_magnitude: np.ndarray) -> Tuple[float, float]:
        """Calculates Otsu optimal threshold and inter-class variance ratio eta."""
        flat = cva_magnitude.flatten()
        counts, bin_edges = np.histogram(flat, bins=256, range=(0.0, float(np.max(flat)) + 1e-4))
        p = counts / float(flat.size)

        omega = np.cumsum(p)
        mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) / 2.0)
        mu_t = mu[-1]

        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-8)
        max_idx = int(np.argmax(sigma_b_sq))

        threshold = float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0)
        # Total variance
        total_var = float(np.var(flat)) + 1e-8
        eta = float(sigma_b_sq[max_idx] / total_var)
        eta = np.clip(eta, 0.0, 1.0)

        # Baseline clamp to avoid thresholding non-existent changes
        safe_threshold = max(threshold, float(np.mean(flat) + 1.2 * np.std(flat)), 0.22)
        return safe_threshold, float(eta)

    @classmethod
    def _extract_polygon(
        cls,
        binary: np.ndarray,
        offset_y: int,
        offset_x: int,
        total_h: int,
        total_w: int,
    ) -> List[PolygonPoint]:
        eroded = ndimage.binary_erosion(binary)
        boundary = binary & (~eroded)
        ys, xs = np.where(boundary)

        if len(ys) == 0:
            h_sub, w_sub = binary.shape
            return [
                PolygonPoint(x=round(offset_x / total_w, 4), y=round(offset_y / total_h, 4)),
                PolygonPoint(x=round((offset_x + w_sub) / total_w, 4), y=round(offset_y / total_h, 4)),
                PolygonPoint(x=round((offset_x + w_sub) / total_w, 4), y=round((offset_y + h_sub) / total_h, 4)),
                PolygonPoint(x=round(offset_x / total_w, 4), y=round((offset_y + h_sub) / total_h, 4)),
            ]

        cy, cx = np.mean(ys), np.mean(xs)
        angles = np.arctan2(ys - cy, xs - cx)
        sort_order = np.argsort(angles)
        step = max(1, len(sort_order) // 20)

        pts = []
        for idx in sort_order[::step]:
            norm_y = round((offset_y + float(ys[idx])) / total_h, 4)
            norm_x = round((offset_x + float(xs[idx])) / total_w, 4)
            pts.append(PolygonPoint(x=norm_x, y=norm_y))
        return pts

    @classmethod
    def render_change_overlay(
        cls,
        cva_magnitude: np.ndarray,
        threshold: float,
    ) -> Image.Image:
        """Renders an RGBA change overlay highlighting changed areas with dynamic flame gradient."""
        norm = np.clip((cva_magnitude - threshold) / (threshold * 1.5 + 1e-6), 0.0, 1.0)
        h, w = cva_magnitude.shape

        mask = cva_magnitude > threshold
        r = np.zeros((h, w), dtype=np.uint8)
        g = np.zeros((h, w), dtype=np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)
        a = np.zeros((h, w), dtype=np.uint8)

        # High change: Neon Amber to Ruby Red
        r[mask] = 239
        g[mask] = (140 - norm[mask] * 100).astype(np.uint8)
        b[mask] = 0
        a[mask] = np.clip(140 + norm[mask] * 100, 140, 240).astype(np.uint8)

        return Image.fromarray(np.stack([r, g, b, a], axis=-1), mode="RGBA")

    @staticmethod
    def _get_quadrant(y: float, x: float) -> str:
        ns = "North" if y < 0.5 else "South"
        ew = "West" if x < 0.5 else "East"
        if 0.35 <= y <= 0.65 and 0.35 <= x <= 0.65:
            return "Central Sector"
        return f"{ns}-{ew}"
