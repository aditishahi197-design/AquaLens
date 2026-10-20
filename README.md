# AquaLens — ML Water Quality Classification Project

Student project scaffold for training and testing ML models on the Kaggle Water Potability dataset.

## Important limitation

The original CSV is **not bundled in this ZIP** because the source dataset's current download terms/access need to be confirmed directly on Kaggle. The project includes a downloader and exact manual instructions. Do not treat this project as a certified water safety device. A model prediction is not proof that water is safe to drink.

## Dataset

Kaggle: https://www.kaggle.com/datasets/adityakadiwal/water-potability

Expected CSV filename: `water_potability.csv`
Expected columns:
`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability`

The commonly circulated version has 3,276 rows, 9 numeric input features, a binary `Potability` target, and missing values in some feature columns. Verify the downloaded file and current Kaggle license/terms before use or redistribution.

## Setup on macOS (VS Code / Terminal)

1. Download the dataset from Kaggle after signing in. Extract `water_potability.csv` into `data/raw/`.
2. Open Terminal and `cd` into the unzipped `AquaLens_ML_Project` folder.
3. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run the data check:

   ```bash
   python src/check_dataset.py
   ```

5. Train and evaluate a Random Forest baseline:

   ```bash
   python src/train_model.py
   ```

6. Test a sample row with the saved model:

   ```bash
   python src/test_prediction.py
   ```

7. Run comprehensive error analysis and diagnostics:

   ```bash
   python src/error_analysis.py
   ```

8. Execute the automated test suite (46 tests):

   ```bash
   python tests/run_tests.py
   ```

Outputs are saved in `models/` and `reports/`.

## Optional Kaggle API download

Use this only if you have configured Kaggle API credentials and accepted the dataset's terms:

```bash
pip install kaggle
kaggle datasets download -d adityakadiwal/water-potability -p data/raw --unzip
```

Then rename the extracted CSV to `water_potability.csv` if necessary.

## What the code does

- Checks the expected columns and dataset shape.
- Uses a stratified train/test split.
- Imputes missing numeric values using medians within a scikit-learn Pipeline (fit only on training data).
- Trains a Random Forest classifier.
- Reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, classification report, and confusion matrix.
- Saves the fitted pipeline and feature importance plot.

## Hardware integration note

This dataset includes features that ordinary low-cost sensors cannot all measure directly (for example, chloramines, sulfate, organic carbon and trihalomethanes). The included model expects **all nine dataset features**. Do not feed it only pH, turbidity and temperature or invent missing values. A later hardware phase must either use compatible measurements for the trained feature set or train and validate a separate model on a dataset matching the available sensor inputs.

## Project safety and scope

This is an academic ML demonstration only. The dataset labels are not a substitute for laboratory testing or compliance with drinking-water standards. Do not use predictions to decide whether water is safe to drink.
