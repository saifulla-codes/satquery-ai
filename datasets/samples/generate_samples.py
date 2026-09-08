"""Synthetic Remote Sensing Sample Generator.

Generates realistic 4-band GeoTIFFs, multitemporal pairs, and optical-SAR pairs
with genuine geospatial tags for instant offline testing and SIH demonstrations.
"""

import os
import struct
import numpy as np
from PIL import Image


def generate_all_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating realistic satellite samples in: {output_dir}")

    # 1. Coastal Port Multispectral GeoTIFF (512x512, 4 Bands: R, G, B, NIR)
    generate_coastal_multispectral_geotiff(
        os.path.join(output_dir, "coastal_port_multispectral.tif")
    )

    # 2. Multitemporal Urban Expansion Pair (512x512 RGB)
    generate_temporal_pair(
        os.path.join(output_dir, "urban_expansion_2021.png"),
        os.path.join(output_dir, "urban_expansion_2025.png"),
    )

    # 3. Optical + SAR Co-registered Pair (512x512)
    generate_optical_sar_pair(
        os.path.join(output_dir, "industrial_harbor_optical.png"),
        os.path.join(output_dir, "industrial_harbor_sar.png"),
    )

    print("Successfully generated all sample satellite imagery datasets!")


def generate_coastal_multispectral_geotiff(filepath: str):
    """Creates a 4-band (R, G, B, NIR) GeoTIFF with water, vegetation, and port structures."""
    h, w = 512, 512
    np.random.seed(42)

    # Coordinates grid
    y, x = np.mgrid[0:h, 0:w]

    # Water body in Western & Southern area
    water_boundary = 220 + 70 * np.sin(x / 40.0) + 30 * np.cos(y / 60.0)
    water_mask = y > water_boundary

    # Vegetated hills in Eastern sector
    veg_mask = (x > 300) & (y < 350) & (~water_mask)

    # Port and built-up area in Central-Northern zone
    port_mask = (x >= 120) & (x <= 320) & (y >= 80) & (y <= 240) & (~water_mask)

    # Band 1: Red (665 nm)
    red = np.zeros((h, w), dtype=np.uint8)
    red[water_mask] = np.random.randint(15, 30, size=np.sum(water_mask))
    red[veg_mask] = np.random.randint(25, 45, size=np.sum(veg_mask))  # Low red (chlorophyll absorbs)
    red[port_mask] = np.random.randint(140, 190, size=np.sum(port_mask))  # High red (concrete)
    # Remaining bare soil/sparse
    other_mask = ~(water_mask | veg_mask | port_mask)
    red[other_mask] = np.random.randint(80, 120, size=np.sum(other_mask))

    # Band 2: Green (560 nm)
    green = np.zeros((h, w), dtype=np.uint8)
    green[water_mask] = np.random.randint(35, 60, size=np.sum(water_mask))
    green[veg_mask] = np.random.randint(90, 140, size=np.sum(veg_mask))  # High green reflectance
    green[port_mask] = np.random.randint(130, 180, size=np.sum(port_mask))
    green[other_mask] = np.random.randint(70, 100, size=np.sum(other_mask))

    # Band 3: Blue (490 nm)
    blue = np.zeros((h, w), dtype=np.uint8)
    blue[water_mask] = np.random.randint(80, 140, size=np.sum(water_mask))  # Strong blue scatter
    blue[veg_mask] = np.random.randint(20, 40, size=np.sum(veg_mask))
    blue[port_mask] = np.random.randint(125, 175, size=np.sum(port_mask))
    blue[other_mask] = np.random.randint(60, 85, size=np.sum(other_mask))

    # Band 4: NIR (842 nm) - Sentinel-2 Band 8
    nir = np.zeros((h, w), dtype=np.uint8)
    nir[water_mask] = np.random.randint(2, 10, size=np.sum(water_mask))   # Near-zero NIR (water absorbs)
    nir[veg_mask] = np.random.randint(180, 245, size=np.sum(veg_mask))   # Very high NIR (spongy mesophyll)
    nir[port_mask] = np.random.randint(110, 160, size=np.sum(port_mask))
    nir[other_mask] = np.random.randint(60, 95, size=np.sum(other_mask))

    # Stack into 4-channel image
    rgba_or_4band = np.stack([red, green, blue, nir], axis=-1)

    # Save as multi-frame TIFF with GeoTIFF tags
    im = Image.fromarray(rgba_or_4band, mode="RGBA")

    # Set up GeoTIFF Tags:
    # 33550: ModelPixelScaleTag = (10.0, 10.0, 0.0) -> 10m GSD
    # 33922: ModelTiepointTag = (0, 0, 0, 83.21, 17.68, 0) -> Visakhapatnam Port
    # 34735: GeoKeyDirectoryTag = (1, 1, 0, 1, 1024, 0, 1, 2)
    tiffinfo = {
        33550: (10.0, 10.0, 0.0),
        33922: (0.0, 0.0, 0.0, 83.218, 17.689, 0.0),
        34735: (1, 1, 0, 1, 1024, 0, 1, 2, 2048, 0, 1, 4326),
        270: "SatQuery AI - Sentinel-2 Multispectral 10m (R, G, B, NIR)",
    }

    im.save(filepath, format="TIFF", tiffinfo=tiffinfo)
    print(f"Generated 4-band GeoTIFF: {filepath}")


def generate_temporal_pair(path_2021: str, path_2025: str):
    """Generates before (2021) and after (2025) imagery demonstrating deforestation & urban growth."""
    h, w = 512, 512
    np.random.seed(101)

    y, x = np.mgrid[0:h, 0:w]

    # River running diagonally
    river = np.abs(y - (0.4 * x + 180 + 20 * np.sin(x / 30.0))) < 18

    # 2021 Base Scene: Mostly green agricultural fields and woodland
    r2021 = np.full((h, w), 55, dtype=np.uint8) + np.random.randint(-10, 10, (h, w), dtype=np.int8)
    g2021 = np.full((h, w), 140, dtype=np.uint8) + np.random.randint(-15, 15, (h, w), dtype=np.int8)
    b2021 = np.full((h, w), 45, dtype=np.uint8) + np.random.randint(-10, 10, (h, w), dtype=np.int8)

    # River colors in 2021
    r2021[river] = 20
    g2021[river] = 50
    b2021[river] = 110

    # Small village in 2021
    village_2021 = (x > 380) & (y > 380)
    r2021[village_2021] = 150
    g2021[village_2021] = 140
    b2021[village_2021] = 130

    r2021_u = np.clip(r2021, 0, 255).astype(np.uint8)
    g2021_u = np.clip(g2021, 0, 255).astype(np.uint8)
    b2021_u = np.clip(b2021, 0, 255).astype(np.uint8)

    img2021 = Image.fromarray(np.stack([r2021_u, g2021_u, b2021_u], axis=-1), mode="RGB")
    img2021.save(path_2021, "PNG")

    # 2025 Scene: Massive industrial park in Northern-Central sector
    r2025 = r2021_u.copy().astype(np.int16)
    g2025 = g2021_u.copy().astype(np.int16)
    b2025 = b2021_u.copy().astype(np.int16)

    # Industrial development zone (cleared vegetation, concrete warehouses, tarmac)
    dev_zone = (x >= 80) & (x <= 340) & (y >= 40) & (y <= 240) & (~river)

    # Concrete buildings (light gray/white)
    buildings = dev_zone & (((x // 30) % 2 == 0) & ((y // 30) % 2 == 0))
    r2025[buildings] = 210
    g2025[buildings] = 215
    b2025[buildings] = 220

    # Asphalt access roads & parking (dark gray)
    roads = dev_zone & (~buildings)
    r2025[roads] = 85
    g2025[roads] = 88
    b2025[roads] = 90

    r2025_u = np.clip(r2025, 0, 255).astype(np.uint8)
    g2025_u = np.clip(g2025, 0, 255).astype(np.uint8)
    b2025_u = np.clip(b2025, 0, 255).astype(np.uint8)

    img2025 = Image.fromarray(np.stack([r2025_u, g2025_u, b2025_u], axis=-1), mode="RGB")
    img2025.save(path_2025, "PNG")

    print(f"Generated temporal pair: {path_2021} and {path_2025}")


def generate_optical_sar_pair(path_optical: str, path_sar: str):
    """Generates optical RGB and Sentinel-1 C-band SAR backscatter image with speckle."""
    h, w = 512, 512
    np.random.seed(777)

    y, x = np.mgrid[0:h, 0:w]

    # Coastal water on right side
    water = x > 300

    # Port docks and ships
    docks = (x >= 240) & (x <= 300) & (y >= 100) & (y <= 420)
    ships = ((x >= 350) & (x <= 390) & (y >= 160) & (y <= 210)) | \
            ((x >= 410) & (x <= 440) & (y >= 290) & (y <= 330))

    # City buildings on left
    urban = (x < 240)

    # 1. Optical RGB Image
    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    # Ocean
    r[water] = 25
    g[water] = 65
    b[water] = 135

    # Urban
    urban_noise_r = np.random.randint(-15, 15, (h, w), dtype=np.int16)
    urban_noise_g = np.random.randint(-15, 15, (h, w), dtype=np.int16)
    urban_noise_b = np.random.randint(-15, 15, (h, w), dtype=np.int16)

    r[urban] = np.clip(145 + urban_noise_r[urban], 0, 255).astype(np.uint8)
    g[urban] = np.clip(140 + urban_noise_g[urban], 0, 255).astype(np.uint8)
    b[urban] = np.clip(135 + urban_noise_b[urban], 0, 255).astype(np.uint8)

    # Concrete docks
    r[docks] = 180
    g[docks] = 180
    b[docks] = 185

    # Cargo vessels
    r[ships] = 190
    g[ships] = 40
    b[ships] = 40

    img_opt = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
    img_opt.save(path_optical, "PNG")

    # 2. SAR C-Band Microwave Backscatter with Speckle Noise
    # Radar intensity: Water = very dark (specular reflection away from sensor: -25 dB)
    # Steel ships & cranes = intense double-bounce reflector (> 0 dB)
    # City buildings = moderate double-bounce (-5 to -10 dB)
    # Bare ground / tarmac = (-15 dB)
    sar_intensity = np.zeros((h, w), dtype=np.float32)

    sar_intensity[water] = 0.02  # ~ -17 dB
    sar_intensity[urban] = 0.35  # ~ -4.5 dB
    sar_intensity[docks] = 0.55  # ~ -2.5 dB
    sar_intensity[ships] = 1.8   # ~ +2.5 dB (Bright corner reflector)

    # Add Rayleigh/Gamma multiplicative speckle noise
    speckle = np.random.exponential(scale=1.0, size=(h, w))
    sar_noisy = sar_intensity * speckle

    # Scale to 8-bit grayscale for image file storage
    sar_norm = np.clip(sar_noisy / 1.5 * 255.0, 0, 255).astype(np.uint8)

    img_sar = Image.fromarray(sar_norm, mode="L")
    img_sar.save(path_sar, "PNG")

    print(f"Generated Optical-SAR pair: {path_optical} and {path_sar}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_samples(current_dir)
