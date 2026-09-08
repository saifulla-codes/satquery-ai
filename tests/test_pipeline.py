"""Comprehensive test suite verifying all core geospatial, spectral, CV, agentic, and killer query capabilities."""

import os
import unittest
import numpy as np
from backend.geospatial.raster.geotiff_reader import GeoTIFFReader, GeoRaster
from backend.geospatial.spectral.spectral_analyzer import SpectralAnalyzer
from backend.models.grounding.spatial_grounder import SpatialGrounder
from backend.models.change_detection.change_detector import ChangeDetector
from backend.models.multimodal.optical_sar_analyzer import OpticalSarAnalyzer
from backend.core.agent.tool_registry import ToolRegistry
from backend.core.agent.planner import TaskPlanner
from backend.core.agent.query_parser import QueryParser, QueryIntent
from backend.services.providers.provider_router import ProviderRouter
from backend.core.agent.orchestrator import AgentOrchestrator
from datasets.base_adapter import BaseDatasetAdapter
from datasets.bigearthnet.adapter import BigEarthNetAdapter
from datasets.vrsbench.adapter import VRSBenchAdapter
from datasets.rsvqa.adapter import RSVQAAdapter, CDVQAAdapter


class TestHardenedSatQueryAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(test_dir)
        cls.samples_dir = os.path.join(project_dir, "datasets", "samples")
        cls.layers_dir = os.path.join(project_dir, "backend", "generated_layers")
        cls.orchestrator = AgentOrchestrator(layers_dir=cls.layers_dir)

    def test_geotiff_metadata_and_bands(self):
        tif_path = os.path.join(self.samples_dir, "coastal_port_multispectral.tif")
        raster = GeoTIFFReader.read(tif_path)

        self.assertEqual(raster.height, 512)
        self.assertEqual(raster.width, 512)
        self.assertEqual(raster.channels, 4)
        self.assertTrue(raster.has_nir)
        self.assertEqual(raster.metadata["data_type"], "uint8")
        self.assertIn("EPSG:4326", raster.crs)
        self.assertIsNotNone(raster.metadata.get("bounds"))
        self.assertEqual(raster.metadata["pixel_size_meters"], 10.0)

        # Test custom band mapping
        custom_mapping = {"red": 0, "green": 1, "blue": 2, "nir": 3}
        raster_mapped = GeoTIFFReader.read(tif_path, band_mapping=custom_mapping)
        self.assertEqual(raster_mapped.band_mapping, custom_mapping)
        print("\n[PASS] GeoTIFF Reader & Metadata: 512x512, 4-bands, CRS=EPSG:4326, GSD=10m")

    def test_ndvi_exact_formula_and_safety(self):
        # Synthetic known raster: 100x100
        h, w = 100, 100
        synthetic_arr = np.zeros((h, w, 4), dtype=np.float32)
        # Red = 0.2, NIR = 0.8 -> Expected NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.60
        synthetic_arr[:, :, 0] = 0.2 * 255.0  # Red
        synthetic_arr[:, :, 1] = 0.4 * 255.0  # Green
        synthetic_arr[:, :, 2] = 0.1 * 255.0  # Blue
        synthetic_arr[:, :, 3] = 0.8 * 255.0  # NIR

        meta = {"width": w, "height": h, "channels": 4, "has_nir": True, "pixel_size_meters": 10.0}
        raster = GeoRaster(array=synthetic_arr, filepath="synthetic.tif", metadata=meta)

        ndvi, metrics, lim = SpectralAnalyzer.calculate_ndvi(raster)
        self.assertIsNone(lim)
        # Verify exact mathematical value within floating tolerance
        self.assertAlmostEqual(metrics["mean_ndvi"], 0.60, places=2)
        self.assertEqual(metrics["dense_vegetation_pct"], 100.0)
        self.assertIn("median_ndvi", metrics)
        self.assertIn("std_ndvi", metrics)
        print(f"[PASS] NDVI Exact Formula: Expected=0.60, Calculated={metrics['mean_ndvi']:.2f}")

    def test_ndvi_rgb_fallback(self):
        # Standard RGB image without NIR
        p_rgb = os.path.join(self.samples_dir, "urban_expansion_2021.png")
        raster_rgb = GeoTIFFReader.read(p_rgb)
        self.assertFalse(raster_rgb.has_nir)

        vari, metrics, limitations = SpectralAnalyzer.calculate_ndvi(raster_rgb)
        self.assertIsNotNone(limitations, "Should explicitly report limitation when NIR is absent")
        self.assertIn("VARI Proxy", metrics["method"])
        print("[PASS] NDVI RGB Fallback: Appropriately triggers VARI and reports sensor limitations")

    def test_water_detection_and_ndwi(self):
        tif_path = os.path.join(self.samples_dir, "coastal_port_multispectral.tif")
        raster = GeoTIFFReader.read(tif_path)

        ndwi, metrics, lim = SpectralAnalyzer.calculate_ndwi(raster)
        self.assertIn("water_coverage_pct", metrics)
        self.assertIn("water_area_hectares", metrics)
        self.assertGreater(metrics["water_coverage_pct"], 50.0)
        print(f"[PASS] Water Detection NDWI: Water={metrics['water_coverage_pct']}% (~{metrics['water_area_hectares']} ha)")

    def test_spatial_grounding_polygon_and_coordinates(self):
        tif_path = os.path.join(self.samples_dir, "coastal_port_multispectral.tif")
        raster = GeoTIFFReader.read(tif_path)

        features, mask, stats = SpatialGrounder.ground_feature(raster, "water")
        self.assertGreater(len(features), 0)
        first_feat = features[0]
        self.assertIsNotNone(first_feat.polygon, "Must extract polygon contour points")
        self.assertGreaterEqual(len(first_feat.polygon), 4, "Polygon must have >= 4 vertices")

        # Verify pixel bounds and geographic bounds
        self.assertIsNotNone(first_feat.pixel_bounds)
        self.assertIn("ymin_px", first_feat.pixel_bounds)
        self.assertIn("xmin_px", first_feat.pixel_bounds)
        self.assertIsNotNone(first_feat.geo_bounds, "Georeferenced raster must produce geo_bounds")
        self.assertIn("west", first_feat.geo_bounds)
        self.assertIn("north", first_feat.geo_bounds)

        # Verify normalized coordinates [0.0 - 1.0]
        for pt in first_feat.polygon:
            self.assertGreaterEqual(pt.x, 0.0)
            self.assertLessEqual(pt.x, 1.0)
            self.assertGreaterEqual(pt.y, 0.0)
            self.assertLessEqual(pt.y, 1.0)
        print(f"[PASS] Spatial Grounding: {len(first_feat.polygon)} polygon vertices, Pixel bounds={first_feat.pixel_bounds}")

    def test_change_detection_with_otsu_separability(self):
        p2021 = os.path.join(self.samples_dir, "urban_expansion_2021.png")
        p2025 = os.path.join(self.samples_dir, "urban_expansion_2025.png")

        raster_b = GeoTIFFReader.read(p2021)
        raster_a = GeoTIFFReader.read(p2025)

        cva, features, metrics, lim = ChangeDetector.detect_changes(raster_b, raster_a)
        self.assertGreater(metrics["total_change_pct"], 2.0)
        self.assertIn("otsu_separability_eta", metrics)
        self.assertGreaterEqual(metrics["otsu_separability_eta"], 0.0)
        self.assertLessEqual(metrics["otsu_separability_eta"], 1.0)
        self.assertGreater(len(features), 0)
        self.assertIn("confidence_method", metrics)
        print(f"[PASS] Change Detection CVA: Change={metrics['total_change_pct']}%, Otsu η={metrics['otsu_separability_eta']}")

    def test_optical_sar_processing(self):
        p_opt = os.path.join(self.samples_dir, "industrial_harbor_optical.png")
        p_sar = os.path.join(self.samples_dir, "industrial_harbor_sar.png")

        raster_opt = GeoTIFFReader.read(p_opt)
        raster_sar = GeoTIFFReader.read(p_sar)

        features, summary, insights = OpticalSarAnalyzer.fuse_optical_sar(raster_opt, raster_sar)
        self.assertGreater(len(features), 0)
        sar_m = summary["sar_metrics"]
        self.assertIn("mean_backscatter_db", sar_m)
        self.assertIn("metallic_builtup_scatter_pct", sar_m)
        self.assertIsNotNone(features[0].pixel_bounds)
        print(f"[PASS] Optical-SAR Fusion: {len(features)} structures, SAR mean={sar_m['mean_backscatter_db']} dB")

    def test_tool_registry_execution_and_properties(self):
        tool = ToolRegistry.get("geotiff_decoder")
        self.assertIsNotNone(tool)
        self.assertIn("filepath", tool.inputs)
        self.assertIn("raster", tool.outputs)
        self.assertIn("OPTICAL", tool.required_modality)

        res = ToolRegistry.execute("geotiff_decoder", filepath=os.path.join(self.samples_dir, "urban_expansion_2021.png"))
        self.assertEqual(res["status"], "completed")
        self.assertGreater(res["duration_ms"], 0.0)
        self.assertIsInstance(res["result"], GeoRaster)
        print(f"[PASS] ToolRegistry: Successfully executed '{tool.name}' in {res['duration_ms']}ms")

    def test_agent_planner_compound_dag(self):
        killer_q = "Identify areas where vegetation decreased while built-up development increased between the two dates, and show the affected regions."
        q_info = QueryParser.parse(killer_q)
        self.assertEqual(q_info["primary_intent"], QueryIntent.COMPOUND_CHANGE)

        steps = TaskPlanner.plan(q_info, is_multitemporal=True)
        tool_names = [s.tool_name for s in steps]
        self.assertIn("query_parser", tool_names)
        self.assertIn("geotiff_decoder", tool_names)
        self.assertIn("temporal_aligner", tool_names)
        self.assertIn("cva_detector", tool_names)
        self.assertIn("ndvi_analyzer", tool_names)
        self.assertIn("builtup_analyzer", tool_names)
        self.assertIn("spatial_intersection", tool_names)
        self.assertIn("vqa_reasoner", tool_names)
        self.assertEqual(len(steps), 8)
        print(f"[PASS] TaskPlanner Compound DAG: 8 specialized steps planned for killer query: {tool_names}")

    def test_killer_query_orchestration(self):
        p2021 = os.path.join(self.samples_dir, "urban_expansion_2021.png")
        p2025 = os.path.join(self.samples_dir, "urban_expansion_2025.png")
        killer_q = "Identify areas where vegetation decreased while built-up development increased between the two dates, and show the affected regions."

        resp = self.orchestrator.analyze_change(p2021, p2025, query=killer_q)
        self.assertEqual(resp.analysis_mode, "LOCAL ANALYSIS")
        self.assertIsNotNone(resp.confidence_method)
        self.assertIn("Otsu", resp.confidence_method)
        self.assertGreater(len(resp.pipeline_steps), 0)
        self.assertEqual(len(resp.execution_trace), len(resp.pipeline_steps))
        self.assertEqual(len(resp.evidence), len(resp.visual_evidence))
        self.assertGreater(len(resp.bounding_boxes), 0)
        # Check compound conversion labeling
        categories = [b.properties.get("category", "") for b in resp.bounding_boxes]
        self.assertTrue(any("Vegetation Converted" in cat for cat in categories))
        print(f"[PASS] Killer Query Orchestration: Successfully executed compound DAG, {resp.change_percentage}% change, Mode={resp.analysis_mode}")

    def test_provider_router_offline_integrity(self):
        active = ProviderRouter.get_active_provider()
        self.assertTrue(active.is_available)
        self.assertEqual(active.name, "local_deterministic_geospatial_engine")
        self.assertEqual(active.provider_type, "LOCAL ANALYSIS")
        providers = ProviderRouter.list_providers()
        self.assertGreaterEqual(len(providers), 5)
        print(f"[PASS] ProviderRouter: Verified active provider '{active.name}' reports mode '{active.provider_type}'")

    def test_dataset_adapters_architecture(self):
        adapters = [
            BigEarthNetAdapter(),
            VRSBenchAdapter(),
            RSVQAAdapter(),
            CDVQAAdapter(),
        ]
        for adp in adapters:
            self.assertIsInstance(adp, BaseDatasetAdapter)
            self.assertIsNotNone(adp.name)
            self.assertIsNotNone(adp.dataset_type)
            # When unmounted, must report is_available == False rather than returning misleading mock data
            self.assertFalse(adp.is_available)
            self.assertEqual(adp.list_samples(), [])
        print("[PASS] Dataset Adapters: All 4 benchmark adapters conform to BaseDatasetAdapter contract")


if __name__ == "__main__":
    unittest.main()
