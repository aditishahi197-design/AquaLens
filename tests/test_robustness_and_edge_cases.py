"""
Robustness, boundary condition, and edge case tests for the AquaLens ML pipeline.
"""

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aqualens_random_forest.joblib"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


class TestRobustnessAndEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Model file missing.")
        cls.model = joblib.load(MODEL_PATH)
        cls.base_values = {
            "ph": 7.0,
            "Hardness": 150.0,
            "Solids": 12000.0,
            "Chloramines": 6.0,
            "Sulfate": 300.0,
            "Conductivity": 400.0,
            "Organic_carbon": 10.0,
            "Trihalomethanes": 60.0,
            "Turbidity": 3.0
        }

    def test_pipeline_handles_single_nan_in_ph(self):
        """Test pipeline median imputation recovers missing pH."""
        sample = dict(self.base_values)
        sample["ph"] = np.nan
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)

        self.assertIn(int(pred[0]), [0, 1])
        self.assertEqual(len(proba[0]), 2)
        self.assertFalse(np.isnan(proba).any())

    def test_pipeline_handles_single_nan_in_sulfate(self):
        """Test pipeline median imputation recovers missing Sulfate."""
        sample = dict(self.base_values)
        sample["Sulfate"] = np.nan
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)

        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_pipeline_handles_multiple_nans(self):
        """Test pipeline handles samples with multiple missing features."""
        sample = dict(self.base_values)
        sample["ph"] = np.nan
        sample["Sulfate"] = np.nan
        sample["Trihalomethanes"] = np.nan
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)

        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_pipeline_handles_all_nans(self):
        """Test pipeline safely handles a row where all 9 features are NaN."""
        df_all_nan = pd.DataFrame([{f: np.nan for f in FEATURES}], columns=FEATURES)

        pred = self.model.predict(df_all_nan)
        proba = self.model.predict_proba(df_all_nan)

        self.assertIn(int(pred[0]), [0, 1])
        self.assertEqual(proba.shape, (1, 2))
        self.assertAlmostEqual(float(np.sum(proba[0])), 1.0, places=5)

    def test_boundary_ph_zero(self):
        """Test boundary pH = 0.0 executes without numerical error."""
        sample = dict(self.base_values)
        sample["ph"] = 0.0
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)
        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_boundary_ph_fourteen(self):
        """Test boundary pH = 14.0 executes without numerical error."""
        sample = dict(self.base_values)
        sample["ph"] = 14.0
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)
        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_all_zeros_input(self):
        """Test pipeline handles all-zero feature vector without zero-division."""
        sample = {f: 0.0 for f in FEATURES}
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)
        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_extreme_scale_solids(self):
        """Test extreme outlier scale values (e.g., 1,000,000 TDS) do not overflow."""
        sample = dict(self.base_values)
        sample["Solids"] = 1_000_000.0
        sample["Conductivity"] = 50_000.0
        df = pd.DataFrame([sample], columns=FEATURES)

        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)
        self.assertIn(int(pred[0]), [0, 1])
        self.assertFalse(np.isnan(proba).any())

    def test_input_immutability(self):
        """Test pipeline prediction does not alter input DataFrame in-place."""
        df_original = pd.DataFrame([self.base_values], columns=FEATURES)
        df_copy = df_original.copy(deep=True)

        _ = self.model.predict(df_original)
        pd.testing.assert_frame_equal(df_original, df_copy)


if __name__ == "__main__":
    unittest.main()
