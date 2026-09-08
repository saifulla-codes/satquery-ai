"""Pure Python & PIL/NumPy GeoTIFF and remote sensing raster parser.

Reads TIFF tags, IFD geokeys (ModelPixelScale, ModelTiepoint, GeoKeyDirectory),
dimensions, band arrangements, data types, and spatial coordinate bounds without heavy C dependencies.
Supports custom band mappings, 8-bit / 16-bit / float32 dynamic range normalization,
and gracefully handles non-georeferenced imagery.
"""

import os
import struct
import numpy as np
from PIL import Image, TiffImagePlugin
from typing import Dict, Any, Optional, Tuple, List


# Standard GeoTIFF Tag IDs
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_SAMPLES_PER_PIXEL = 277
TAG_PLANAR_CONFIG = 284
TAG_SAMPLE_FORMAT = 339
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_MODEL_TRANSFORMATION = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737
TAG_GDAL_METADATA = 42112
TAG_GDAL_NODATA = 42113


class GeoRaster:
    """Represents a loaded satellite raster image with spectral and geospatial metadata."""

    def __init__(
        self,
        array: np.ndarray,
        filepath: str,
        metadata: Dict[str, Any],
        geo_transform: Optional[Tuple[float, float, float, float, float, float]] = None,
        crs: Optional[str] = None,
        band_mapping: Optional[Dict[str, int]] = None,
    ):
        self.array = array  # Shape: (H, W, C) or (H, W)
        self.filepath = filepath
        self.metadata = metadata
        self.geo_transform = geo_transform  # (west, pixel_w, 0, north, 0, -pixel_h)
        self.crs = crs
        # Default band mapping: Red=0, Green=1, Blue=2, NIR=3 if >=4 channels
        self.band_mapping = band_mapping or self._default_band_mapping()

    def _default_band_mapping(self) -> Dict[str, int]:
        c = self.channels
        if c >= 4:
            return {"red": 0, "green": 1, "blue": 2, "nir": 3}
        elif c == 3:
            return {"red": 0, "green": 1, "blue": 2}
        elif c == 2:
            return {"red": 0, "green": 1}
        else:
            return {"red": 0}

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def channels(self) -> int:
        return self.array.shape[2] if self.array.ndim > 2 else 1

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)

    @property
    def has_nir(self) -> bool:
        return "nir" in self.band_mapping and self.band_mapping["nir"] < self.channels

    def get_band(self, name: str) -> Optional[np.ndarray]:
        """Returns normalized float32 array [0.0 - 1.0] for the requested band name."""
        idx = self.band_mapping.get(name.lower())
        if idx is None or idx >= self.channels:
            return None

        if self.array.ndim == 2:
            raw_band = self.array.astype(np.float32)
        else:
            raw_band = self.array[:, :, idx].astype(np.float32)

        # Handle NoData if configured
        nodata = self.metadata.get("nodata_value")
        if nodata is not None:
            raw_band = np.where(raw_band == nodata, np.nan, raw_band)

        return self._normalize_band(raw_band)

    def get_rgb(self) -> np.ndarray:
        """Returns standard 8-bit RGB array (H, W, 3) for browser rendering with 2%-98% stretch."""
        r_idx = self.band_mapping.get("red", 0)
        g_idx = self.band_mapping.get("green", min(1, self.channels - 1))
        b_idx = self.band_mapping.get("blue", min(2, self.channels - 1))

        if self.array.ndim == 2 or self.channels == 1:
            band_raw = self.array if self.array.ndim == 2 else self.array[:, :, 0]
            gray = self._stretch_to_uint8(band_raw)
            return np.stack([gray, gray, gray], axis=-1)

        r = self._stretch_to_uint8(self.array[:, :, r_idx])
        g = self._stretch_to_uint8(self.array[:, :, g_idx])
        b = self._stretch_to_uint8(self.array[:, :, b_idx])
        return np.stack([r, g, b], axis=-1)

    def get_nir(self) -> Optional[np.ndarray]:
        """Returns normalized float NIR band [0.0 - 1.0] if present."""
        return self.get_band("nir")

    def get_red(self) -> np.ndarray:
        """Returns normalized float Red band [0.0 - 1.0]."""
        band = self.get_band("red")
        if band is None:
            band = self._normalize_band(self.array[:, :, 0].astype(np.float32) if self.array.ndim > 2 else self.array.astype(np.float32))
        return band

    def get_green(self) -> np.ndarray:
        """Returns normalized float Green band [0.0 - 1.0]."""
        band = self.get_band("green")
        if band is None:
            return self.get_red()
        return band

    def get_blue(self) -> np.ndarray:
        """Returns normalized float Blue band [0.0 - 1.0]."""
        band = self.get_band("blue")
        if band is None:
            return self.get_red()
        return band

    @staticmethod
    def _normalize_band(arr: np.ndarray) -> np.ndarray:
        """Normalizes band to [0.0, 1.0] range respecting data type scale."""
        valid_mask = ~np.isnan(arr)
        if not np.any(valid_mask):
            return np.zeros_like(arr, dtype=np.float32)

        max_val = float(np.nanmax(arr[valid_mask]))
        if max_val <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)

        if max_val <= 1.0:
            return np.clip(arr, 0.0, 1.0).astype(np.float32)
        elif max_val <= 255.0:
            return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)
        elif max_val <= 10000.0:  # Sentinel-2 BOA reflectance scaled by 10000
            return np.clip(arr / 10000.0, 0.0, 1.0).astype(np.float32)
        elif max_val <= 65535.0:  # 16-bit unsigned
            return np.clip(arr / 65535.0, 0.0, 1.0).astype(np.float32)
        else:
            return np.clip(arr / max_val, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Applies robust 2% - 98% linear percentile stretch for optimal visual dynamic range."""
        arr_f = arr.astype(np.float32)
        valid = arr_f[~np.isnan(arr_f)]
        if len(valid) == 0:
            return np.zeros(arr.shape, dtype=np.uint8)

        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        if p98 > p2:
            stretched = (arr_f - p2) / (p98 - p2) * 255.0
            return np.clip(stretched, 0, 255).astype(np.uint8)
        max_v, min_v = np.max(valid), np.min(valid)
        if max_v > min_v:
            return np.clip((arr_f - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)
        return np.zeros(arr.shape, dtype=np.uint8)


class GeoTIFFReader:
    """Extracts raw pixels, bands, and geospatial tags from GeoTIFF and common raster formats."""

    @classmethod
    def read(cls, filepath: str, band_mapping: Optional[Dict[str, int]] = None) -> GeoRaster:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Raster file not found: {filepath}")

        # Magic byte inspection for file format verification
        format_detected = cls._detect_format_by_magic(filepath)

        metadata: Dict[str, Any] = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "format": format_detected,
            "has_nir": False,
            "modality": "OPTICAL",
        }

        # Check for SAR naming indicators or metadata tags
        basename_upper = os.path.basename(filepath).upper()
        if any(k in basename_upper for k in ["SAR", "SENTINEL1", "S1", "BACKSCATTER", "RADAR"]):
            metadata["modality"] = "SAR"

        with Image.open(filepath) as img:
            width, height = img.size

            geo_transform = None
            crs = None
            bounds = None
            pixel_size_m = None
            nodata_val = None

            # Extract TIFF-specific geokeys if TIFF/GeoTIFF
            if hasattr(img, "tag_v2") and img.tag_v2 is not None:
                tags = img.tag_v2
                metadata["tiff_tags_detected"] = True

                # Check ModelPixelScale (33550) & ModelTiepoint (33922)
                scale = tags.get(TAG_MODEL_PIXEL_SCALE)
                tiepoint = tags.get(TAG_MODEL_TIEPOINT)

                if scale and tiepoint and len(scale) >= 2 and len(tiepoint) >= 6:
                    scale_x, scale_y = float(scale[0]), float(scale[1])
                    tp_i, tp_j, tp_x, tp_y = (
                        float(tiepoint[0]),
                        float(tiepoint[1]),
                        float(tiepoint[3]),
                        float(tiepoint[4]),
                    )

                    origin_x = tp_x - (tp_i * scale_x)
                    origin_y = tp_y + (tp_j * scale_y)

                    geo_transform = (origin_x, scale_x, 0.0, origin_y, 0.0, -scale_y)
                    pixel_size_m = scale_x

                    # Compute bounding box coordinates
                    west = origin_x
                    north = origin_y
                    east = origin_x + (width * scale_x)
                    south = origin_y - (height * scale_y)
                    bounds = {
                        "west": round(west, 6),
                        "south": round(south, 6),
                        "east": round(east, 6),
                        "north": round(north, 6),
                    }

                # GeoKeyDirectory (34735) for CRS
                geokeys = tags.get(TAG_GEO_KEY_DIRECTORY)
                if geokeys:
                    crs = cls._parse_geokeys(geokeys)

                # GDAL NoData (42113)
                nodata_tag = tags.get(TAG_GDAL_NODATA)
                if nodata_tag:
                    try:
                        nodata_val = float(str(nodata_tag).strip("\x00"))
                    except (ValueError, TypeError):
                        pass

            # Multi-frame TIFF (each band as frame) or standard array
            if getattr(img, "n_frames", 1) > 1 and format_detected in ["TIFF", "GEOTIFF"]:
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frames.append(np.array(img))
                array = np.stack(frames, axis=-1)
            else:
                array = np.array(img)

            channels = array.shape[2] if array.ndim > 2 else 1
            metadata["channels"] = channels
            metadata["width"] = width
            metadata["height"] = height
            metadata["data_type"] = str(array.dtype)
            metadata["nodata_value"] = nodata_val
            metadata["crs"] = crs or ("EPSG:4326 (WGS 84)" if bounds else None)
            metadata["bounds"] = bounds
            metadata["pixel_size_meters"] = pixel_size_m

            # Band names and semantics
            if channels >= 4:
                metadata["has_nir"] = True
                metadata["modality"] = "MULTISPECTRAL"
                metadata["band_names"] = ["Red", "Green", "Blue", "Near-Infrared (NIR)"]
                if channels > 4:
                    for extra in range(5, channels + 1):
                        metadata["band_names"].append(f"Band {extra}")
            elif channels == 3:
                metadata["band_names"] = ["Red", "Green", "Blue"]
            elif channels == 1:
                metadata["band_names"] = ["Intensity / Radar Backscatter" if metadata["modality"] == "SAR" else "Band 1"]

            return GeoRaster(
                array=array,
                filepath=filepath,
                metadata=metadata,
                geo_transform=geo_transform,
                crs=crs,
                band_mapping=band_mapping,
            )

    @staticmethod
    def _detect_format_by_magic(filepath: str) -> str:
        """Verifies binary magic bytes at start of file."""
        try:
            with open(filepath, "rb") as f:
                header = f.read(8)
                if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
                    return "TIFF"
                elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                    return "PNG"
                elif header.startswith(b"\xff\xd8\xff"):
                    return "JPEG"
        except Exception:
            pass
        ext = os.path.splitext(filepath)[1].lower().replace(".", "")
        return ext.upper() if ext else "UNKNOWN"

    @staticmethod
    def _parse_geokeys(geokeys: Tuple[int, ...]) -> str:
        """Parses GeoKeyDirectory array to extract EPSG / Projected / Geographic CRS code."""
        if len(geokeys) < 4:
            return "EPSG:4326 (WGS 84)"
        num_keys = geokeys[3]
        for idx in range(1, num_keys + 1):
            offset = idx * 4
            if offset + 3 < len(geokeys):
                key_id = geokeys[offset]
                val_offset = geokeys[offset + 3]
                if key_id == 3072:  # ProjectedCSTypeGeoKey
                    return f"EPSG:{val_offset} (Projected CRS)"
                elif key_id == 2048:  # GeographicTypeGeoKey
                    return f"EPSG:{val_offset} (Geographic CRS)"
        return "EPSG:4326 (WGS 84)"
