
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "water_potability.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise SystemExit(f"Missing columns: {missing_columns}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)

    if y.nunique() != 2:
        raise SystemExit("Expected target classes 0 and 1.")

    # Keep a separate test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        ))
    ])

    # Compare Random Forest settings using training data only.
    parameter_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 15],
        "classifier__min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("Running five-fold cross-validation and Random Forest tuning...")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    # Best estimator is refitted on the full training set by GridSearchCV.
    model = search.best_estimator_

    # Evaluate the selected model once on the held-out test set.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "random_state": 42,
        "test_size": 0.20,
        "cross_validation_folds": 5,
        "cross_validation_scoring": "balanced_accuracy",
        "best_cross_validation_balanced_accuracy": float(search.best_score_),
        "best_parameters": search.best_params_,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "class_distribution_full_data": {
            str(k): int(v)
            for k, v in y.value_counts().sort_index().items()
        },
        "note": (
            "Academic benchmark only. Not validated for drinking-water "
            "safety or physical sensor deployment."
        )
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the filename used by the existing Streamlit application.
    joblib.dump(model, MODEL_DIR / "aqualens_random_forest.joblib")

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Class 0", "Class 1"],
            zero_division=0
        )
    )

    # Confusion matrix.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Class 0", "Class 1"]
    )
    disp.plot()
    plt.title("AquaLens Random Forest - Held-out Test Set")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Feature importance.
    classifier = model.named_steps["classifier"]
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    ax = importance_df.sort_values("importance").plot.barh(
        x="feature", y="importance", legend=False
    )
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance (not causation)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    # Save the tuning results for your project documentation.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(REPORT_DIR / "random_forest_tuning_results.csv", index=False)

    print("\nTraining and evaluation completed.")
    print("\nBest Random Forest parameters:")
    print(search.best_params_)
    print(
        "Best 5-fold CV balanced accuracy:",
        round(search.best_score_, 4)
    )

    print("\nHeld-out test metrics:")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("\nSaved model:", MODEL_DIR / "aqualens_random_forest.joblib")
    print("Saved reports:", REPORT_DIR)


if __name__ == "__main__":
    main()
