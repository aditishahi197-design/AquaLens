"""
AquaLens Inference and Validation Module

Core validation, prediction, and report-generation logic decoupled
from the Streamlit UI presentation layer for automated testing and reuse.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

FEATURES: List[str] = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

DEFAULT_VALUES: Dict[str, float] = {
    "ph": 7.0,
    "Hardness": 150.0,
    "Solids": 12000.0,
    "Chloramines": 6.0,
    "Sulfate": 300.0,
    "Conductivity": 400.0,
    "Organic_carbon": 10.0,
    "Trihalomethanes": 60.0,
    "Turbidity": 3.0,
}

PARAMETER_NAMES: Dict[str, str] = {
    "ph": "pH",
    "Hardness": "Hardness",
    "Solids": "Total dissolved solids",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Conductivity",
    "Organic_carbon": "Organic carbon",
    "Trihalomethanes": "Trihalomethanes",
    "Turbidity": "Turbidity",
}

LABELS: Dict[int, str] = {
    0: "Not Potable (Class 0)",
    1: "Potable (Class 1)",
}


def validate_sample(sample: Any) -> pd.DataFrame:
    """
    Validates a single-sample DataFrame for inference.

    Raises:
        ValueError: If input structure, types, counts, or values violate domain rules.
    """
    if not isinstance(sample, pd.DataFrame):
        raise ValueError("Input data is invalid: expected pandas DataFrame.")

    if list(sample.columns) != FEATURES:
        raise ValueError(
            f"The nine required features are missing or out of order. "
            f"Expected {FEATURES}, got {list(sample.columns)}."
        )

    if len(sample) != 1:
        raise ValueError("Please analyze one sample at a time.")

    numeric_sample = sample.apply(pd.to_numeric, errors="coerce")

    if numeric_sample.isna().any().any():
        raise ValueError("All nine measurements must be valid numbers.")

    arr = numeric_sample.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Measurements must be finite numbers.")

    values = numeric_sample.iloc[0]

    if not 0.0 <= values["ph"] <= 14.0:
        raise ValueError(f"pH must be between 0 and 14. Got {values['ph']}.")

    if (values < 0.0).any():
        negative_cols = values[values < 0.0].index.tolist()
        raise ValueError(
            f"Measurements cannot be negative. Negative values in: {negative_cols}."
        )

    return numeric_sample


def predict_sample(model: Any, sample: pd.DataFrame) -> Tuple[int, List[int], np.ndarray]:
    """
    Validates input and runs inference using the trained pipeline.

    Returns:
        (prediction, classes, probabilities)
    """
    validated_sample = validate_sample(sample)

    prediction = int(model.predict(validated_sample)[0])
    probabilities = model.predict_proba(validated_sample)[0]
    classes = [int(value) for value in model.classes_]

    return prediction, classes, probabilities


def format_metric(value: Any) -> str:
    """Formats a float metric as a percentage string, or returns 'N/A'."""
    if not isinstance(value, (int, float, np.floating, np.integer)) or not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}"


def create_prediction_report(sample: pd.DataFrame, result: Dict[str, Any], timestamp: str) -> str:
    """Generates an academic/audit report string from prediction outputs."""
    prediction = result["prediction"]
    classes = result["classes"]
    probabilities = result["probabilities"]

    predicted_label = LABELS.get(
        prediction,
        f"Unknown class ({prediction})",
    )

    if prediction == 1:
        interpretation = (
            "The model classified this sample as POTABLE according to "
            "patterns learned from the training dataset."
        )
    else:
        interpretation = (
            "The model classified this sample as NOT POTABLE according "
            "to patterns learned from the training dataset."
        )

    lines = [
        "AQUALENS - WATER QUALITY PREDICTION REPORT",
        "=" * 52,
        "",
        f"Report generated: {timestamp}",
        "Model: Random Forest Classifier",
        "",
        "1. PREDICTION OUTPUT",
        "-" * 30,
        f"Predicted class number: {prediction}",
        f"Predicted label: {predicted_label}",
        f"Interpretation: {interpretation}",
        "",
        "2. CLASS DEFINITIONS",
        "-" * 30,
        "Class 0: Not potable according to the dataset label.",
        "Class 1: Potable according to the dataset label.",
        "",
        "3. CLASS PROBABILITIES",
        "-" * 30,
    ]

    for class_label, probability in zip(classes, probabilities):
        label = LABELS.get(class_label, f"Class {class_label}")
        lines.append(f"{label}: {float(probability) * 100:.2f}%")

    lines.extend([
        "",
        "4. INPUT WATER MEASUREMENTS",
        "-" * 30,
    ])

    for feature in FEATURES:
        name = PARAMETER_NAMES.get(feature, feature)
        value = float(sample.iloc[0][feature])
        lines.append(f"{name}: {value:.4f}")

    lines.extend([
        "",
        "5. IMPORTANT LIMITATIONS",
        "-" * 30,
        "This is a machine-learning prediction based on dataset labels.",
        "Class probabilities are model estimates, not laboratory results.",
        "This report does not certify drinking-water safety.",
        "Use appropriate laboratory testing and applicable water-quality",
        "standards before making decisions about drinking water.",
        "",
        "End of report",
    ])

    return "\n".join(lines)
