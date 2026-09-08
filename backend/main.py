"""SatQuery AI — Hardened REST API Backend.

Serves satellite imagery ingestion, geospatial metadata extraction,
agentic orchestration, multitemporal change detection, SAR fusion, and analysis caching.
"""

import os
import io
import uuid
from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from PIL import Image
from pydantic import ValidationError

from backend.api.schemas import (
    AnalysisRequest,
    ChangeDetectionRequest,
    OpticalSarRequest,
    GeoMetadata,
)
from backend.geospatial.raster.geotiff_reader import GeoTIFFReader
from backend.core.agent.orchestrator import AgentOrchestrator
from backend.core.agent.tool_registry import ToolRegistry
from backend.services.analysis_service import AnalysisService
from backend.services.providers.provider_router import ProviderRouter

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
LAYERS_DIR = os.path.join(BASE_DIR, "generated_layers")
CACHE_DIR = os.path.join(BASE_DIR, "analysis_cache")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "datasets", "samples")
FRONTEND_DIST = os.path.join(PROJECT_DIR, "frontend", "dist")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(LAYERS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Initialize Services
AnalysisService.init(CACHE_DIR)
orchestrator = AgentOrchestrator(layers_dir=LAYERS_DIR)

app = Flask(__name__)
max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024  # Default 100 MB max upload limit


# Global CORS Handler
@app.after_request
def add_cors_headers(response):
    allowed_origins_env = os.getenv("CORS_ORIGINS", "*")
    req_origin = request.headers.get("Origin")
    if allowed_origins_env == "*":
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif req_origin and req_origin in [o.strip() for o in allowed_origins_env.split(",")]:
        response.headers["Access-Control-Allow-Origin"] = req_origin
    else:
        response.headers["Access-Control-Allow-Origin"] = allowed_origins_env.split(",")[0].strip() if allowed_origins_env else "*"

    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return make_response("", 200)


@app.errorhandler(400)
def bad_request_error(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400


@app.errorhandler(404)
def not_found_error(e):
    return jsonify({"error": "Resource Not Found", "message": str(e)}), 404


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@app.route("/", defaults={"path": ""}, methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def serve_frontend_or_404(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    if path != "" and os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"status": "SatQuery AI Backend Online", "frontend_build": "Ready"})


@app.route("/api/health", methods=["GET"])
def health_check():
    """System health, hardware capabilities, active providers, and specialist tools."""
    active_provider = ProviderRouter.get_active_provider()
    return jsonify({
        "status": "online",
        "service": "SatQuery AI Remote Sensing Assistant",
        "version": "1.2.0",
        "competition": "Smart India Hackathon 2026 (SIH26167)",
        "organization": "Indian Space Research Organisation (ISRO)",
        "platform": "Apple Silicon ARM64 / M2 16GB",
        "active_reasoning_provider": active_provider.name,
        "active_analysis_mode": active_provider.provider_type,
        "specialist_tools_count": len(ToolRegistry.list_tools()),
        "specialist_tools": ToolRegistry.list_tools(),
    })


@app.route("/api/providers", methods=["GET"])
def list_providers():
    """Lists supported AI model providers and their current availability."""
    return jsonify({
        "active_provider": ProviderRouter.get_active_provider().name,
        "active_mode": ProviderRouter.get_active_provider().provider_type,
        "providers": ProviderRouter.list_providers(),
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Lists curated ready-to-use sample satellite datasets for instant SIH demonstrations."""
    samples = [
        {
            "id": "sample_coastal_multispectral",
            "filename": "coastal_port_multispectral.tif",
            "title": "Visakhapatnam Port & Estuary (Synthetic Sentinel-2 4-Band GeoTIFF Fixture)",
            "modality": "MULTISPECTRAL",
            "bands": ["Red", "Green", "Blue", "Near-Infrared (NIR)"],
            "resolution": "10.0m (Sentinel-2 GSD)",
            "crs": "EPSG:4326 (WGS 84)",
            "description": "Calibrated synthetic 4-band multispectral satellite scene with marine water, vegetated hills, and shipping docks.",
            "is_synthetic_fixture": True,
            "preset_queries": [
                "Describe this region and classify visible land cover.",
                "Where are the water bodies located?",
                "Where is vegetation concentrated? Calculate NDVI.",
                "Highlight the shipping port and industrial facilities.",
            ],
            "image_url": "/api/images/coastal_port_multispectral.tif",
        },
        {
            "id": "sample_urban_2021",
            "filename": "urban_expansion_2021.png",
            "title": "Industrial Expansion: T1 Baseline (Synthetic Optical Fixture)",
            "modality": "OPTICAL",
            "bands": ["Red", "Green", "Blue"],
            "resolution": "10.0m",
            "description": "Pre-development rural landscape showing natural river channel, agricultural vegetation, and village.",
            "is_synthetic_fixture": True,
            "preset_queries": ["What is the baseline land use in this 2021 capture?"],
            "image_url": "/api/images/urban_expansion_2021.png",
            "paired_temporal_id": "sample_urban_2025",
        },
        {
            "id": "sample_urban_2025",
            "filename": "urban_expansion_2025.png",
            "title": "Industrial Expansion: T2 Evolution (Synthetic Optical Fixture)",
            "modality": "OPTICAL",
            "bands": ["Red", "Green", "Blue"],
            "resolution": "10.0m",
            "description": "Post-development acquisition showing large logistics park, cleared vegetation, and new concrete warehouses.",
            "is_synthetic_fixture": True,
            "preset_queries": [
                "What changed between the 2021 and 2025 captures?",
                "Identify areas where vegetation decreased while built-up development increased between the two dates, and show the affected regions.",
            ],
            "image_url": "/api/images/urban_expansion_2025.png",
            "paired_temporal_id": "sample_urban_2021",
        },
        {
            "id": "sample_harbor_optical",
            "filename": "industrial_harbor_optical.png",
            "title": "Harbor Complex (Synthetic High-Resolution Optical Fixture)",
            "modality": "OPTICAL",
            "bands": ["Red", "Green", "Blue"],
            "resolution": "10.0m",
            "description": "High-contrast visible spectrum acquisition showing urban skyline, marine dock infrastructure, and vessels.",
            "is_synthetic_fixture": True,
            "preset_queries": ["Identify maritime vessels and coastal structures."],
            "image_url": "/api/images/industrial_harbor_optical.png",
            "paired_sar_id": "sample_harbor_sar",
        },
        {
            "id": "sample_harbor_sar",
            "filename": "industrial_harbor_sar.png",
            "title": "Harbor Complex (Synthetic Sentinel-1 C-Band SAR Fixture)",
            "modality": "SAR",
            "bands": ["C-Band Backscatter Intensity"],
            "resolution": "10.0m",
            "description": "Microwave radar backscatter with speckle noise, bright metallic corner reflectors, and dark specular water return.",
            "is_synthetic_fixture": True,
            "preset_queries": ["Analyze radar backscatter to verify reinforced structures."],
            "image_url": "/api/images/industrial_harbor_sar.png",
            "paired_optical_id": "sample_harbor_optical",
        },
    ]
    return jsonify(samples)


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles raw satellite imagery upload, verifies binary magic bytes, parses metadata, and registers for analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file field in multipart upload"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported format '{ext}'. Allowed extensions: {allowed_exts}"}), 400

    image_id = f"img_{uuid.uuid4().hex[:10]}"
    saved_filename = f"{image_id}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)
    file.save(saved_path)

    # Inspect magic bytes
    try:
        with open(saved_path, "rb") as f:
            header = f.read(8)
            is_valid_tiff = header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")
            is_valid_png = header.startswith(b"\x89PNG\r\n\x1a\n")
            is_valid_jpeg = header.startswith(b"\xff\xd8\xff")

            if ext in [".tif", ".tiff"] and not is_valid_tiff:
                os.remove(saved_path)
                return jsonify({"error": "Corrupted or invalid TIFF header"}), 400
            elif ext == ".png" and not is_valid_png:
                os.remove(saved_path)
                return jsonify({"error": "Corrupted or invalid PNG header"}), 400
            elif ext in [".jpg", ".jpeg"] and not is_valid_jpeg:
                os.remove(saved_path)
                return jsonify({"error": "Corrupted or invalid JPEG header"}), 400

        raster = GeoTIFFReader.read(saved_path)
        metadata = raster.metadata
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({"error": f"Failed to parse raster: {str(e)}"}), 400

    return jsonify({
        "image_id": image_id,
        "filename": file.filename,
        "url": f"/api/images/{saved_filename}",
        "metadata": metadata,
        "message": "Satellite image ingested, verified, and parsed successfully",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Executes single-image Visual Question Answering, Grounding, or Spectral Analysis."""
    data = request.get_json(force=True, silent=True)
    if not data or "image_id" not in data or "query" not in data:
        return jsonify({"error": "Missing required fields 'image_id' or 'query'"}), 400

    image_id = data["image_id"]
    query = data["query"]
    task_hint = data.get("task_hint")
    band_mapping = data.get("band_mapping")

    filepath = _resolve_image_path(image_id)
    if not filepath:
        return jsonify({"error": f"Image '{image_id}' not found"}), 404

    try:
        response = orchestrator.analyze_single_image(filepath, query, task_hint, band_mapping)
        return jsonify(response.model_dump())
    except Exception as e:
        return jsonify({"error": f"Analysis pipeline failed: {str(e)}"}), 500


@app.route("/api/change-detection", methods=["POST"])
def change_detection():
    """Executes multitemporal change detection between T1 (before) and T2 (after) images."""
    data = request.get_json(force=True, silent=True)
    if not data or "image_before_id" not in data or "image_after_id" not in data:
        return jsonify({"error": "Requires 'image_before_id' and 'image_after_id'"}), 400

    p_before = _resolve_image_path(data["image_before_id"])
    p_after = _resolve_image_path(data["image_after_id"])

    if not p_before or not p_after:
        return jsonify({"error": "One or both images could not be located"}), 404

    query = data.get("query", "What changed between these two images?")
    try:
        response = orchestrator.analyze_change(p_before, p_after, query)
        return jsonify(response.model_dump())
    except Exception as e:
        return jsonify({"error": f"Change detection failed: {str(e)}"}), 500


@app.route("/api/optical-sar", methods=["POST"])
def optical_sar_fusion():
    """Executes cross-modal fusion between Optical and SAR imagery."""
    data = request.get_json(force=True, silent=True)
    if not data or "optical_image_id" not in data or "sar_image_id" not in data:
        return jsonify({"error": "Requires 'optical_image_id' and 'sar_image_id'"}), 400

    p_opt = _resolve_image_path(data["optical_image_id"])
    p_sar = _resolve_image_path(data["sar_image_id"])

    if not p_opt or not p_sar:
        return jsonify({"error": "One or both images could not be located"}), 404

    query = data.get("query", "Compare optical and SAR evidence to identify development.")
    try:
        response = orchestrator.analyze_optical_sar(p_opt, p_sar, query)
        return jsonify(response.model_dump())
    except Exception as e:
        return jsonify({"error": f"Optical-SAR analysis failed: {str(e)}"}), 500


@app.route("/api/analysis/<analysis_id>", methods=["GET"])
def get_analysis_by_id(analysis_id):
    """Retrieves cached analysis results by analysis_id."""
    record = AnalysisService.get_analysis(analysis_id)
    if not record:
        return jsonify({"error": f"Analysis ID '{analysis_id}' not found"}), 404
    return jsonify(record)


@app.route("/api/analysis", methods=["GET"])
def list_analyses():
    """Lists recent analysis runs."""
    recent = AnalysisService.list_recent()
    return jsonify({"count": len(recent), "analyses": recent})


@app.route("/api/images/<filename>", methods=["GET"])
def serve_image(filename):
    """Serves browser-compatible RGB representation of an uploaded or sample image."""
    filepath = _find_file_by_name(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Image file not found"}), 404

    ext = os.path.splitext(filename)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return send_file(filepath, mimetype=f"image/{ext.replace('.', '')}")

    # Convert GeoTIFF or multi-band TIFF to RGB PNG in-memory with percentile stretch
    try:
        raster = GeoTIFFReader.read(filepath)
        rgb = raster.get_rgb()
        pil_img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to render raster: {str(e)}"}), 500


@app.route("/api/layers/<filename>", methods=["GET"])
def serve_layer(filename):
    """Serves generated visual evidence overlay layer (transparent RGBA PNG)."""
    layer_path = os.path.join(LAYERS_DIR, filename)
    if not os.path.exists(layer_path):
        return jsonify({"error": "Layer not found"}), 404
    return send_file(layer_path, mimetype="image/png")


def _resolve_image_path(identifier: str) -> str:
    """Resolves image ID or filename to full local path."""
    for fname in os.listdir(SAMPLES_DIR):
        if identifier in fname or fname == identifier:
            return os.path.join(SAMPLES_DIR, fname)

    for fname in os.listdir(UPLOADS_DIR):
        if identifier in fname or fname == identifier:
            return os.path.join(UPLOADS_DIR, fname)

    id_map = {
        "sample_coastal_multispectral": "coastal_port_multispectral.tif",
        "sample_urban_2021": "urban_expansion_2021.png",
        "sample_urban_2025": "urban_expansion_2025.png",
        "sample_harbor_optical": "industrial_harbor_optical.png",
        "sample_harbor_sar": "industrial_harbor_sar.png",
    }
    if identifier in id_map:
        return os.path.join(SAMPLES_DIR, id_map[identifier])

    return None


def _find_file_by_name(filename: str) -> str:
    for folder in [SAMPLES_DIR, UPLOADS_DIR]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Hardened SatQuery AI Backend at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
