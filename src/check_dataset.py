from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "water_potability.csv"
EXPECTED = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"
]


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}\n"
            "Download water_potability.csv from Kaggle and place it in data/raw/. "
            "See README.md for instructions."
        )
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")
    print("Dataset loaded successfully")
    print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
    print("\nColumn types:")
    print(df[EXPECTED].dtypes)
    print("\nMissing values:")
    print(df[EXPECTED].isna().sum())
    print("\nTarget distribution (Potability):")
    print(df["Potability"].value_counts(dropna=False).sort_index())
    print("\nPreview:")
    print(df[EXPECTED].head())


if __name__ == "__main__":
    main()
