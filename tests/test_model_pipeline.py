"""
Unit tests for the AquaLens trained scikit-learn Pipeline.
"""

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aqualens_random_forest.joblib"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest(f"Model file missing at: {MODEL_PATH}")
        cls.model = joblib.load(MODEL_PATH)
        cls.sample_df = pd.DataFrame([{
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

    def test_model_file_exists(self):
        """Test model joblib artifact exists."""
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}")

    def test_pipeline_instance(self):
        """Test loaded object is a scikit-learn Pipeline."""
        self.assertIsInstance(self.model, Pipeline)

    def test_pipeline_steps(self):
        """Test pipeline contains 'imputer' and 'classifier' steps."""
        step_names = [name for name, _ in self.model.steps]
        self.assertIn("imputer", step_names)
        self.assertIn("classifier", step_names)
        self.assertIsInstance(self.model.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(self.model.named_steps["classifier"], RandomForestClassifier)

    def test_classes_attribute(self):
        """Test classifier classes are binary [0, 1]."""
        self.assertEqual(list(self.model.classes_), [0, 1])

    def test_single_prediction_shape_and_type(self):
        """Test single prediction returns a valid integer class {0, 1}."""
        pred = self.model.predict(self.sample_df)
        self.assertEqual(len(pred), 1)
        self.assertIn(int(pred[0]), [0, 1])

    def test_predict_proba_format(self):
        """Test predict_proba returns normalized 2-class probability distribution."""
        proba = self.model.predict_proba(self.sample_df)
        self.assertEqual(proba.shape, (1, 2))
        self.assertTrue((proba >= 0.0).all() and (proba <= 1.0).all())
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=5)

    def test_batch_prediction(self):
        """Test pipeline handles batch input with varying rows."""
        batch_df = pd.concat([self.sample_df] * 15, ignore_index=True)
        batch_preds = self.model.predict(batch_df)
        self.assertEqual(len(batch_preds), 15)
        batch_probs = self.model.predict_proba(batch_df)
        self.assertEqual(batch_probs.shape, (15, 2))

    def test_prediction_determinism(self):
        """Test predictions are strictly deterministic for identical inputs."""
        pred_1 = self.model.predict(self.sample_df)
        prob_1 = self.model.predict_proba(self.sample_df)

        pred_2 = self.model.predict(self.sample_df)
        prob_2 = self.model.predict_proba(self.sample_df)

        self.assertEqual(int(pred_1[0]), int(pred_2[0]))
        np.testing.assert_array_almost_equal(prob_1, prob_2, decimal=6)


if __name__ == "__main__":
    unittest.main()
