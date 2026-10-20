"""
Unit tests for AquaLens dataset integrity, schema, types, and domain rules.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "water_potability.csv"

EXPECTED_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"
ALL_COLUMNS = EXPECTED_FEATURES + [TARGET]


class TestDatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_PATH.exists():
            raise unittest.SkipTest(f"Dataset missing at {DATA_PATH}")
        cls.df = pd.read_csv(DATA_PATH)

    def test_file_exists(self):
        """Test dataset CSV exists on disk."""
        self.assertTrue(DATA_PATH.exists(), f"Missing dataset file: {DATA_PATH}")

    def test_row_count(self):
        """Test dataset contains expected number of rows (3276 expected)."""
        self.assertEqual(len(self.df), 3276, f"Unexpected row count: {len(self.df)}")

    def test_column_presence_and_order(self):
        """Test all 10 expected columns exist."""
        self.assertEqual(list(self.df.columns), ALL_COLUMNS)

    def test_numeric_data_types(self):
        """Test all feature and target columns are numeric."""
        for col in ALL_COLUMNS:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.df[col]),
                f"Column {col} is not numeric: {self.df[col].dtype}"
            )

    def test_target_binary_classes(self):
        """Test target Potability has only binary values {0, 1}."""
        unique_targets = set(self.df[TARGET].dropna().unique())
        self.assertEqual(unique_targets, {0, 1})

    def test_target_has_no_missing_values(self):
        """Test target Potability has zero missing values."""
        self.assertEqual(self.df[TARGET].isna().sum(), 0)

    def test_missing_values_locations(self):
        """Test missing values only appear in expected columns (ph, Sulfate, Trihalomethanes)."""
        cols_with_na = self.df.columns[self.df.isna().any()].tolist()
        expected_na_cols = ["ph", "Sulfate", "Trihalomethanes"]
        self.assertEqual(sorted(cols_with_na), sorted(expected_na_cols))

    def test_ph_physical_range(self):
        """Test valid pH values are physically plausible within [0, 14]."""
        valid_ph = self.df["ph"].dropna()
        self.assertTrue((valid_ph >= 0.0).all(), "Found negative pH values")
        self.assertTrue((valid_ph <= 14.0).all(), "Found pH values above 14.0")

    def test_non_negative_physical_parameters(self):
        """Test chemical and physical parameters are strictly non-negative."""
        for feature in [
            "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ]:
            valid_vals = self.df[feature].dropna()
            self.assertTrue(
                (valid_vals >= 0.0).all(),
                f"Found negative values in feature {feature}"
            )


if __name__ == "__main__":
    unittest.main()
