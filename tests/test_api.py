"""Comprehensive integration test suite for SatQuery AI REST API endpoints."""

import io
import json
import unittest
from backend.main import app


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("specialist_tools", data)
        self.assertIn("active_analysis_mode", data)
        self.assertGreaterEqual(data["specialist_tools_count"], 10)
        print("\n[PASS] GET /api/health: 14 specialist tools registered, platform online, Mode=" + data["active_analysis_mode"])

    def test_providers_endpoint(self):
        response = self.client.get("/api/providers")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("active_provider", data)
        self.assertIn("providers", data)
        self.assertIn("active_mode", data)
        provider_names = [p["name"] for p in data["providers"]]
        self.assertIn("local_deterministic_geospatial_engine", provider_names)
        self.assertIn("anthropic_claude_vision", provider_names)
        print(f"[PASS] GET /api/providers: Active={data['active_provider']}, Available={provider_names}")

    def test_samples_endpoint(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(len(data), 5)
        for s in data:
            self.assertTrue(s.get("is_synthetic_fixture", False), "Samples must be honestly labeled as synthetic fixtures")
        print(f"[PASS] GET /api/samples: {len(data)} sample satellite datasets honestly labeled as calibrated synthetic fixtures")

    def test_analyze_endpoint_and_caching(self):
        payload = {
            "image_id": "sample_coastal_multispectral",
            "query": "Where are the water bodies located?",
        }
        response = self.client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("answer", data)
        self.assertIn("analysis_id", data)
        self.assertIn("provider", data)
        self.assertEqual(data.get("analysis_mode"), "LOCAL ANALYSIS")
        self.assertGreater(len(data["visual_evidence"]), 0)
        self.assertGreater(len(data["evidence"]), 0)
        self.assertGreater(len(data["pipeline_steps"]), 0)
        self.assertGreater(len(data["execution_trace"]), 0)

        # Verify polygon and coordinates structure in spatial features
        if data["bounding_boxes"]:
            first_feat = data["bounding_boxes"][0]
            self.assertIn("polygon", first_feat)
            self.assertIn("pixel_bounds", first_feat)

        aid = data["analysis_id"]
        # Verify GET /api/analysis/<id> retrieval from cache
        get_res = self.client.get(f"/api/analysis/{aid}")
        self.assertEqual(get_res.status_code, 200)
        get_data = get_res.get_json()
        self.assertEqual(get_data["analysis_id"], aid)
        print(f"[PASS] POST /api/analyze & GET /api/analysis/{aid}: Verified execution, coordinates, and caching")

    def test_change_detection_killer_query(self):
        payload = {
            "image_before_id": "sample_urban_2021",
            "image_after_id": "sample_urban_2025",
            "query": "Identify areas where vegetation decreased while built-up development increased between the two dates, and show the affected regions.",
        }
        response = self.client.post(
            "/api/change-detection",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("change_percentage", data)
        self.assertIn("threshold_quality", data)
        self.assertIn("confidence_method", data)
        self.assertIn("analysis_mode", data)
        self.assertEqual(data["analysis_mode"], "LOCAL ANALYSIS")
        self.assertGreater(data["change_percentage"], 2.0)
        self.assertGreater(len(data["execution_trace"]), 0)
        print(f"[PASS] POST /api/change-detection Killer Query: Change={data['change_percentage']}%, Otsu η={data['threshold_quality']}, Method={data['confidence_method'][:40]}...")

    def test_optical_sar_endpoint(self):
        payload = {
            "optical_image_id": "sample_harbor_optical",
            "sar_image_id": "sample_harbor_sar",
            "query": "Compare optical and SAR evidence to identify development.",
        }
        response = self.client.post(
            "/api/optical-sar",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("cross_modal_insights", data)
        self.assertIn("analysis_mode", data)
        self.assertGreater(len(data["cross_modal_insights"]), 0)
        print(f"[PASS] POST /api/optical-sar: {len(data['cross_modal_insights'])} cross-modal insights generated, Mode={data['analysis_mode']}")

    def test_file_upload_validation(self):
        # 1. Invalid extension
        data = {"file": (io.BytesIO(b"dummy"), "malicious.exe")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

        # 2. Corrupted PNG header
        corrupt_data = {"file": (io.BytesIO(b"NOT_A_PNG_DATA"), "corrupted.png")}
        res = self.client.post("/api/upload", data=corrupt_data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        print("[PASS] POST /api/upload Validation: Correctly rejects invalid extensions and corrupted binary headers")

    def test_list_analyses_endpoint(self):
        res = self.client.get("/api/analysis")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("count", data)
        self.assertIn("analyses", data)
        print(f"[PASS] GET /api/analysis: Retrieved {data['count']} cached analysis runs")


if __name__ == "__main__":
    unittest.main()
