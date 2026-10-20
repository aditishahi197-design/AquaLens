"""
Performance benchmarking and artifact integrity tests for AquaLens.
"""

from pathlib import Path
import json
import time
import unittest
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aqualens_random_forest.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
ERROR_REPORT_PATH = ROOT / "reports" / "error_analysis_report.json"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


class TestPerformanceAndArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Model file missing.")
        cls.model = joblib.load(MODEL_PATH)
        cls.sample = pd.DataFrame([{
            "ph": 7.0,
            "Hardness": 150.0,
            "Solids": 12000.0,
            "Chloramines": 6.0,
            "Sulfate": 300.0,
            "Conductivity": 400.0,
            "Organic_carbon": 10.0,
            "Trihalomethanes": 60.0,
            "Turbidity": 3.0
        }], columns=FEATURES)

    def test_single_sample_latency(self):
        """Test single sample prediction latency is strictly under 50ms on average."""
        # Warmup
        _ = self.model.predict(self.sample)

        runs = 50
        start = time.perf_counter()
        for _ in range(runs):
            _ = self.model.predict(self.sample)
        total_time = time.perf_counter() - start
        avg_latency = total_time / runs

        self.assertLess(
            avg_latency,
            0.050,
            f"Average single-sample inference too slow: {avg_latency * 1000:.2f}ms (threshold 50ms)"
        )

    def test_batch_sample_latency(self):
        """Test batch prediction of 100 samples executes in under 200ms."""
        batch = pd.concat([self.sample] * 100, ignore_index=True)
        # Warmup
        _ = self.model.predict(batch)

        start = time.perf_counter()
        _ = self.model.predict(batch)
        batch_latency = time.perf_counter() - start

        self.assertLess(
            batch_latency,
            0.200,
            f"Batch inference of 100 samples too slow: {batch_latency * 1000:.2f}ms (threshold 200ms)"
        )

    def test_model_artifact_size(self):
        """Test model artifact is present and under 50MB."""
        size_bytes = MODEL_PATH.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        self.assertGreater(size_mb, 1.0, f"Model file suspiciously small: {size_mb:.2f}MB")
        self.assertLess(size_mb, 50.0, f"Model file exceeds 50MB limit: {size_mb:.2f}MB")

    def test_metrics_json_schema(self):
        """Test metrics.json exists and contains all required metrics."""
        self.assertTrue(METRICS_PATH.exists(), "metrics.json missing")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "accuracy", "balanced_accuracy", "roc_auc",
            "f1_macro", "precision_macro", "recall_macro",
            "best_parameters"
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing required metric key: {key}")
            self.assertIsInstance(data[key], (float, int, dict))

    def test_error_analysis_report_json_schema(self):
        """Test error_analysis_report.json exists and contains detailed sections."""
        self.assertTrue(ERROR_REPORT_PATH.exists(), "error_analysis_report.json missing")
        with open(ERROR_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("overfitting_diagnostic", data)
        self.assertIn("test_metrics", data)
        self.assertIn("missingness_analysis", data)
        self.assertIn("subgroup_analysis", data)
        self.assertIn("confidence_and_calibration", data)
        self.assertIn("threshold_sensitivity", data)
        self.assertIn("recommendations", data)


if __name__ == "__main__":
    unittest.main()
