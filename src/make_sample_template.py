from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
features = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
out = ROOT / "data" / "sample_input_template.csv"
pd.DataFrame([{c: "" for c in features}]).to_csv(out, index=False)
print(f"Created empty input template: {out}")
print("Fill it only with genuine measurements in the expected units. Do not use it for potability decisions.")
