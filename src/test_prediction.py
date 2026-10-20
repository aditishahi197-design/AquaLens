from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aqualens_random_forest.joblib"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Trained model not found. Run: python src/train_model.py")
    # Illustrative example only. These are made-up inputs for testing software flow,
    # not a real sample and not a water-safety recommendation.
    sample = pd.DataFrame([{
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
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict(sample)[0])
    probability = model.predict_proba(sample)[0]
    print("Illustrative software test input:")
    print(sample.to_string(index=False))
    print(f"\nModel class output: {prediction} (dataset label only)")
    print(f"Model class probabilities: class 0={probability[0]:.3f}, class 1={probability[1]:.3f}")
    print("WARNING: This is not a real water sample or a drinking-water safety determination.")


if __name__ == "__main__":
    main()
