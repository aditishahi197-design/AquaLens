"""
Unit tests for input validation, inference orchestration, and reporting logic.
"""

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from src.inference import (
    FEATURES,
    DEFAULT_VALUES,
    validate_sample,
    predict_sample,
    format_metric,
    create_prediction_report
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aqualens_random_forest.joblib"


class TestInferenceAndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Model file not found.")
        cls.model = joblib.load(MODEL_PATH)
        cls.valid_sample = pd.DataFrame([DEFAULT_VALUES], columns=FEATURES)

    def test_valid_sample_passes_validation(self):
        """Test valid default sample passes validation cleanly."""
        validated = validate_sample(self.valid_sample)
        self.assertIsInstance(validated, pd.DataFrame)
        self.assertEqual(len(validated), 1)

    def test_non_dataframe_input_rejected(self):
        """Test non-DataFrame inputs raise ValueError."""
        for invalid in [{"ph": 7.0}, [7.0] * 9, "ph,7.0", None, 123]:
            with self.assertRaises(ValueError, msg=f"Failed to reject: {type(invalid)}"):
                validate_sample(invalid)

    def test_missing_column_rejected(self):
        """Test sample missing any feature column raises ValueError."""
        df_missing = self.valid_sample.drop(columns=["ph"])
        with self.assertRaises(ValueError):
            validate_sample(df_missing)

    def test_wrong_column_order_rejected(self):
        """Test columns in unexpected order raise ValueError."""
        reversed_cols = list(reversed(FEATURES))
        df_reversed = self.valid_sample[reversed_cols]
        with self.assertRaises(ValueError):
            validate_sample(df_reversed)

    def test_multiple_rows_rejected(self):
        """Test batch/multiple rows in single-sample inference raises ValueError."""
        df_multi = pd.concat([self.valid_sample, self.valid_sample], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_sample(df_multi)

    def test_empty_dataframe_rejected(self):
        """Test empty DataFrame raises ValueError."""
        df_empty = pd.DataFrame(columns=FEATURES)
        with self.assertRaises(ValueError):
            validate_sample(df_empty)

    def test_nan_values_rejected(self):
        """Test presence of NaN in user input raises ValueError."""
        df_nan = self.valid_sample.copy()
        df_nan.loc[0, "ph"] = np.nan
        with self.assertRaises(ValueError):
            validate_sample(df_nan)

    def test_inf_values_rejected(self):
        """Test infinite values raise ValueError."""
        df_inf = self.valid_sample.copy()
        df_inf.loc[0, "Solids"] = np.inf
        with self.assertRaises(ValueError):
            validate_sample(df_inf)

    def test_non_numeric_strings_rejected(self):
        """Test non-numeric string values raise ValueError."""
        sample_dict = dict(DEFAULT_VALUES)
        sample_dict["Turbidity"] = "not_a_number"
        df_str = pd.DataFrame([sample_dict], columns=FEATURES)
        with self.assertRaises(ValueError):
            validate_sample(df_str)

    def test_ph_out_of_bounds_rejected(self):
        """Test pH < 0 or > 14 raises ValueError."""
        df_low = self.valid_sample.copy()
        df_low.loc[0, "ph"] = -0.5
        with self.assertRaises(ValueError):
            validate_sample(df_low)

        df_high = self.valid_sample.copy()
        df_high.loc[0, "ph"] = 14.5
        with self.assertRaises(ValueError):
            validate_sample(df_high)

    def test_negative_physical_parameters_rejected(self):
        """Test negative values for physical parameters raise ValueError."""
        for col in ["Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity"]:
            df_neg = self.valid_sample.copy()
            df_neg.loc[0, col] = -1.0
            with self.assertRaises(ValueError, msg=f"Failed to reject negative {col}"):
                validate_sample(df_neg)

    def test_predict_sample_execution(self):
        """Test predict_sample returns correct tuple structure."""
        pred, classes, probs = predict_sample(self.model, self.valid_sample)
        self.assertIn(pred, [0, 1])
        self.assertEqual(classes, [0, 1])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_format_metric_valid_floats(self):
        """Test format_metric converts floats to percentage strings."""
        self.assertEqual(format_metric(0.6585), "65.85%")
        self.assertEqual(format_metric(1.0), "100.00%")
        self.assertEqual(format_metric(0.0), "0.00%")

    def test_format_metric_invalid_inputs(self):
        """Test format_metric handles NaN, inf, None, strings gracefully."""
        self.assertEqual(format_metric(np.nan), "N/A")
        self.assertEqual(format_metric(np.inf), "N/A")
        self.assertEqual(format_metric(None), "N/A")
        self.assertEqual(format_metric("unknown"), "N/A")

    def test_create_prediction_report_contents(self):
        """Test create_prediction_report generates expected sections and warnings."""
        result = {
            "prediction": 1,
            "classes": [0, 1],
            "probabilities": [0.42, 0.58]
        }
        report = create_prediction_report(self.valid_sample, result, "2026-10-20_17-00-00")
        self.assertIn("AQUALENS - WATER QUALITY PREDICTION REPORT", report)
        self.assertIn("1. PREDICTION OUTPUT", report)
        self.assertIn("Predicted class number: 1", report)
        self.assertIn("Potable (Class 1)", report)
        self.assertIn("5. IMPORTANT LIMITATIONS", report)
        self.assertIn("This report does not certify drinking-water safety", report)


if __name__ == "__main__":
    unittest.main()
