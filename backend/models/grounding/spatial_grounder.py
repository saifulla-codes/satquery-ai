"""Spatial Grounding engine: Visual localization, connected components, bounding boxes, and polygon boundary contours."""

import numpy as np
from PIL import Image
from scipy import ndimage
from typing import List, Dict, Any, Tuple, Optional
from ...geospatial.raster.geotiff_reader import GeoRaster
from ...api.schemas import SpatialFeature, PolygonPoint


class SpatialGrounder:
    """Grounds user queries in physical pixel space, generating bounding boxes, masks, and polygon contours."""

    @classmethod
    def ground_feature(
        cls,
        raster: GeoRaster,
        feature_type: str,
        threshold_override: Optional[float] = None,
    ) -> Tuple[List[SpatialFeature], np.ndarray, Dict[str, Any]]:
        """Extracts spatial bounding boxes, polygon contours, and binary mask for requested feature type.

        Supported feature_type: 'water', 'vegetation', 'built_up', 'barren', 'salient'
        """
        h, w = raster.height, raster.width
        feature_type = feature_type.lower()

        # Compute classification mask
        if "water" in feature_type:
            mask, label = cls._segment_water(raster)
        elif any(k in feature_type for k in ["veg", "crop", "forest", "field", "tree"]):
            mask, label = cls._segment_vegetation(raster)
        elif any(k in feature_type for k in ["urban", "build", "struct", "construct", "road", "city", "port"]):
            mask, label = cls._segment_builtup(raster)
        else:
            mask, label = cls._segment_salient(raster)

        # Morphological clean up: remove isolated speckles and close small intra-cluster holes
        structuring_element = ndimage.generate_binary_structure(2, 1)
        cleaned_mask = ndimage.binary_opening(mask, structure=structuring_element, iterations=1)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=structuring_element, iterations=2)

        # Connected component labeling
        labeled_array, num_features = ndimage.label(cleaned_mask)

        features: List[SpatialFeature] = []
        min_cluster_pixels = max(int(h * w * 0.001), 25)  # Min 0.1% of scene
        component_sizes = ndimage.sum(cleaned_mask, labeled_array, range(1, num_features + 1))

        pixel_res_m = raster.metadata.get("pixel_size_meters") or 10.0

        sorted_indices = np.argsort(-np.array(component_sizes))[:10]
        slices = ndimage.find_objects(labeled_array)

        for rank, comp_idx in enumerate(sorted_indices):
            comp_label = comp_idx + 1
            pixel_count = component_sizes[comp_idx]
            if pixel_count < min_cluster_pixels:
                continue

            if comp_idx >= len(slices) or slices[comp_idx] is None:
                continue

            slice_y, slice_x = slices[comp_idx]

            ymin = float(slice_y.start / h)
            xmin = float(slice_x.start / w)
            ymax = float(slice_y.stop / h)
            xmax = float(slice_x.stop / w)

            area_sq_m = float(pixel_count * (pixel_res_m ** 2))
            area_ha = round(area_sq_m / 10000.0, 2)

            center_y = (ymin + ymax) / 2.0
            center_x = (xmin + xmax) / 2.0
            quadrant = cls._get_quadrant(center_y, center_x)

            # Defensible spectral purity confidence: fraction of active feature pixels within bounding slice
            box_mask = cleaned_mask[slice_y, slice_x]
            fill_density = float(np.sum(box_mask) / max(box_mask.size, 1))
            # Defensible confidence based on spatial coherence and cluster size
            conf = round(min(0.70 + fill_density * 0.28, 0.98), 2)

            # Extract polygon boundary points
            comp_binary = (labeled_array[slice_y, slice_x] == comp_label)
            polygon_pts = cls._extract_polygon_contour(
                comp_binary,
                offset_y=slice_y.start,
                offset_x=slice_x.start,
                total_h=h,
                total_w=w,
            )

            pixel_bounds = {
                "ymin_px": int(slice_y.start),
                "xmin_px": int(slice_x.start),
                "ymax_px": int(slice_y.stop),
                "xmax_px": int(slice_x.stop),
            }

            geo_bounds = None
            bounds = raster.metadata.get("bounds")
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
                    id=f"feat_{label.lower().replace(' ', '_')}_{rank+1}",
                    label=f"{label} ({quadrant})",
                    ymin=round(ymin, 4),
                    xmin=round(xmin, 4),
                    ymax=round(ymax, 4),
                    xmax=round(xmax, 4),
                    polygon=polygon_pts,
                    confidence=conf,
                    area_hectares=area_ha if area_ha > 0.01 else None,
                    pixel_bounds=pixel_bounds,
                    geo_bounds=geo_bounds,
                    properties={
                        "pixel_count": int(pixel_count),
                        "quadrant": quadrant,
                        "fill_density": round(fill_density, 3),
                        "vertex_count": len(polygon_pts),
                        "coordinate_reference": "EPSG:4326" if geo_bounds else "Image Pixel Space",
                    },
                )
            )

        total_pixels = h * w
        total_feature_pixels = int(np.sum(cleaned_mask))
        feature_pct = round(total_feature_pixels / total_pixels * 100, 2)
        total_ha = round((total_feature_pixels * (pixel_res_m ** 2)) / 10000.0, 2)

        stats = {
            "feature_label": label,
            "detected_clusters": len(features),
            "coverage_percentage": feature_pct,
            "total_area_hectares": total_ha,
            "total_pixels": total_feature_pixels,
        }

        return features, cleaned_mask, stats

    @classmethod
    def _extract_polygon_contour(
        cls,
        binary_slice: np.ndarray,
        offset_y: int,
        offset_x: int,
        total_h: int,
        total_w: int,
        max_vertices: int = 24,
    ) -> List[PolygonPoint]:
        """Traces boundary contour of a connected component and samples normalized polygon points."""
        eroded = ndimage.binary_erosion(binary_slice)
        boundary = binary_slice & (~eroded)

        ys, xs = np.where(boundary)
        if len(ys) == 0:
            # Fallback to 4 bounding corners if boundary empty
            h_sub, w_sub = binary_slice.shape
            return [
                PolygonPoint(x=round(offset_x / total_w, 4), y=round(offset_y / total_h, 4)),
                PolygonPoint(x=round((offset_x + w_sub) / total_w, 4), y=round(offset_y / total_h, 4)),
                PolygonPoint(x=round((offset_x + w_sub) / total_w, 4), y=round((offset_y + h_sub) / total_h, 4)),
                PolygonPoint(x=round(offset_x / total_w, 4), y=round((offset_y + h_sub) / total_h, 4)),
            ]

        # Radial angle sorting from centroid for smooth polygon ordering
        cy, cx = np.mean(ys), np.mean(xs)
        angles = np.arctan2(ys - cy, xs - cx)
        sort_order = np.argsort(angles)

        ys_sorted = ys[sort_order]
        xs_sorted = xs[sort_order]

        # Downsample uniformly to max_vertices
        step = max(1, len(ys_sorted) // max_vertices)
        sampled_y = ys_sorted[::step]
        sampled_x = xs_sorted[::step]

        pts = []
        for py, px in zip(sampled_y, sampled_x):
            norm_y = round((offset_y + float(py)) / total_h, 4)
            norm_x = round((offset_x + float(px)) / total_w, 4)
            pts.append(PolygonPoint(x=norm_x, y=norm_y))

        return pts

    @classmethod
    def create_overlay_image(
        cls,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (59, 130, 246),
        alpha: int = 160,
    ) -> Image.Image:
        """Generates a transparent RGBA image highlighting the binary mask with edge contouring."""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Interior fill
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = alpha

        # Edge enhancement
        eroded = ndimage.binary_erosion(mask)
        edges = mask & (~eroded)
        rgba[edges, 0] = min(color[0] + 40, 255)
        rgba[edges, 1] = min(color[1] + 40, 255)
        rgba[edges, 2] = min(color[2] + 40, 255)
        rgba[edges, 3] = 240

        return Image.fromarray(rgba, mode="RGBA")

    @classmethod
    def _segment_water(cls, raster: GeoRaster) -> Tuple[np.ndarray, str]:
        if raster.has_nir:
            nir = raster.get_nir()
            green = raster.get_green()
            ndwi = (green - nir) / (green + nir + 1e-6)
            mask = ndwi > 0.05
        else:
            red = raster.get_red()
            green = raster.get_green()
            blue = raster.get_blue()
            mask = (blue > red * 1.15) & (green > red * 1.05) & (red < 0.35)
        return mask, "Water Body"

    @classmethod
    def _segment_vegetation(cls, raster: GeoRaster) -> Tuple[np.ndarray, str]:
        if raster.has_nir:
            nir = raster.get_nir()
            red = raster.get_red()
            ndvi = (nir - red) / (nir + red + 1e-6)
            mask = ndvi > 0.25
        else:
            red = raster.get_red()
            green = raster.get_green()
            blue = raster.get_blue()
            exg = 2.0 * green - red - blue
            mask = (exg > 0.08) & (green > red)
        return mask, "Vegetated Tract"

    @classmethod
    def _segment_builtup(cls, raster: GeoRaster) -> Tuple[np.ndarray, str]:
        red = raster.get_red()
        green = raster.get_green()
        blue = raster.get_blue()
        brightness = (red + green + blue) / 3.0
        spectral_neutrality = 1.0 - (np.abs(red - green) + np.abs(green - blue) + np.abs(blue - red)) / 3.0
        mask = (brightness > 0.4) & (spectral_neutrality > 0.85)

        water_mask, _ = cls._segment_water(raster)
        mask = mask & (~water_mask)
        return mask, "Built-up Structure"

    @classmethod
    def _segment_salient(cls, raster: GeoRaster) -> Tuple[np.ndarray, str]:
        rgb = raster.get_rgb().astype(np.float32) / 255.0
        variance = np.std(rgb, axis=-1)
        mask = variance > np.percentile(variance, 75)
        return mask, "Prominent Land Feature"

    @staticmethod
    def _get_quadrant(y: float, x: float) -> str:
        ns = "North" if y < 0.5 else "South"
        ew = "West" if x < 0.5 else "East"
        if 0.35 <= y <= 0.65 and 0.35 <= x <= 0.65:
            return "Central Sector"
        return f"{ns}-{ew}"
