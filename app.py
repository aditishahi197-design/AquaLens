
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "aqualens_random_forest.joblib"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

FEATURES = [
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

DEFAULT_VALUES = {
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

PARAMETER_NAMES = {
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

LABELS = {
    0: "Not Potable (Class 0)",
    1: "Potable (Class 1)",
}

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("aqualens")

st.set_page_config(
    page_title="AquaLens | Water Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 2. STYLING
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --teal: #087E8B;
    --teal-dark: #06636D;
    --navy: #173849;
    --muted: #778B99;
    --border: #E2EAF0;
    --background: #F4F8FB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--navy);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.2rem 1rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 0 22px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #DDF5F4;
    color: #087E8B;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 22px;
}

.brand-name {
    font-family: 'Manrope', sans-serif;
    color: #123344;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -1px;
}

.brand-subtitle {
    color: #8194A1;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.4px;
}

.sidebar-label {
    color: #91A1AE;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 15px 0 9px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 5px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 13px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #EEF8F9;
}

.sidebar-status {
    padding: 11px;
    border-radius: 9px;
    background: #EAF8F1;
    border: 1px solid #D1EEDF;
    color: #187650;
    font-size: 12px;
    font-weight: 700;
}

.sidebar-detail {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 2px;
    border-bottom: 1px solid #EDF1F4;
    font-size: 11px;
    color: #81909D;
}

.sidebar-detail strong {
    color: #264557;
    font-size: 12px;
}

.hero {
    background: linear-gradient(120deg, #073B4C, #087E8B 70%, #169C98);
    border-radius: 17px;
    padding: 26px 29px;
    color: white;
    margin-bottom: 23px;
}

.hero-status {
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.12);
    padding: 6px 9px;
    border-radius: 6px;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 8px;
    letter-spacing: -0.7px;
}

.hero p {
    color: #E0F5F5;
    font-size: 13px;
    line-height: 1.7;
    margin: 0;
    max-width: 760px;
}

.panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 19px;
    margin-bottom: 15px;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #193849;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: #8293A0;
    font-size: 12px;
    margin-bottom: 14px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #718096;
    font-size: 12px;
}

div[data-testid="stMetricValue"] {
    color: #173B4D;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 19px;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background: var(--teal);
    color: white;
    border: 1px solid var(--teal);
    border-radius: 8px;
    min-height: 41px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--teal-dark);
    color: white;
    border-color: var(--teal-dark);
}

.result-class {
    font-family: 'Manrope', sans-serif;
    color: #087E8B;
    font-size: 27px;
    font-weight: 800;
    margin: 10px 0;
}

.result-label {
    color: #526B7B;
    font-size: 13px;
    line-height: 1.8;
}

.probability-label {
    display: flex;
    justify-content: space-between;
    color: #4D6373;
    font-size: 12px;
    font-weight: 600;
    margin: 13px 0 6px;
}

.progress-track {
    height: 8px;
    border-radius: 10px;
    background: #EAF0F4;
    overflow: hidden;
}

.progress-fill {
    height: 8px;
    border-radius: 10px;
}

.info-box {
    background: #EFF8FA;
    border: 1px solid #D7ECEF;
    border-radius: 9px;
    padding: 12px;
    color: #426575;
    font-size: 12px;
    line-height: 1.7;
    margin: 12px 0;
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 24px;
    padding: 17px 0 6px;
    text-align: center;
    color: #91A0AC;
    font-size: 10px;
    line-height: 1.8;
}

@media (max-width: 768px) {
    .hero { padding: 21px; }
    .hero h1 { font-size: 24px; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND METRICS
# =====================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else None

    except (OSError, ValueError, TypeError):
        logger.exception("Unable to read metrics file.")
        return None


if not MODEL_PATH.exists():
    st.error("The trained model file was not found.")
    st.code("python src/train_model.py")
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        st.error("The saved model does not support the required predictions.")
        st.stop()

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != FEATURES:
            st.error(
                "The model's input features do not match the application."
            )
            st.stop()

except Exception:
    logger.exception("Model loading failed.")
    st.error(
        "The model could not be loaded. Check the model file and "
        "installed Python packages."
    )
    st.stop()

metrics = load_metrics()


# =====================================================
# 4. SESSION STATE
# =====================================================

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "last_sample" not in st.session_state:
    st.session_state["last_sample"] = None

if "last_prediction_result" not in st.session_state:
    st.session_state["last_prediction_result"] = None

if "last_report_timestamp" not in st.session_state:
    st.session_state["last_report_timestamp"] = None


def go_to_analyze():
    st.session_state["nav_page"] = "Analyze Sample"


# =====================================================
# 5. NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">A</div>
        <div>
            <div class="brand-name">AquaLens</div>
            <div class="brand-subtitle">WATER INTELLIGENCE</div>
        </div>
    </div>
    <div class="sidebar-label">WORKSPACE</div>
    """, unsafe_allow_html=True)

    st.radio(
        "Navigation",
        [
            "Overview",
            "Analyze Sample",
            "Model Reports",
            "Prediction History",
        ],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-label">SYSTEM STATUS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-status">Model loaded successfully</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="sidebar-detail">
        <span>Algorithm</span>
        <strong>Random Forest Classifier</strong>
    </div>
    <div class="sidebar-detail">
        <span>Input features</span>
        <strong>9 water parameters</strong>
    </div>
    <div class="sidebar-detail">
        <span>Dataset size</span>
        <strong>3,276 samples</strong>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 6. SHARED FUNCTIONS
# =====================================================

def page_header(title, description):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-status">AQUALENS</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def predict_sample(sample):
    if not isinstance(sample, pd.DataFrame):
        raise ValueError("Input data is invalid.")

    if list(sample.columns) != FEATURES:
        raise ValueError(
            "The nine required features are missing or out of order."
        )

    if len(sample) != 1:
        raise ValueError("Please analyze one sample at a time.")

    sample = sample.apply(pd.to_numeric, errors="coerce")

    if sample.isna().any().any():
        raise ValueError("All nine measurements must be valid numbers.")

    if not np.isfinite(sample.to_numpy(dtype=float)).all():
        raise ValueError("Measurements must be finite numbers.")

    values = sample.iloc[0]

    if not 0 <= values["ph"] <= 14:
        raise ValueError("pH must be between 0 and 14.")

    if (values < 0).any():
        raise ValueError("Measurements cannot be negative.")

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    classes = [int(value) for value in model.classes_]

    return prediction, classes, probabilities


def format_metric(value):
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return "N/A"

    return f"{value:.2%}"


def create_prediction_report(sample, result, timestamp):
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


def show_prediction(sample, result, timestamp):
    prediction = result["prediction"]
    classes = result["classes"]
    probabilities = result["probabilities"]

    predicted_label = LABELS.get(
        prediction,
        f"Unknown class ({prediction})",
    )

    if prediction == 1:
        interpretation = (
            "The model classified this sample as POTABLE based on "
            "patterns learned from the training dataset."
        )
    else:
        interpretation = (
            "The model classified this sample as NOT POTABLE based "
            "on patterns learned from the training dataset."
        )

    st.markdown("### Prediction output")

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Predicted result</div>
        <div class="result-class">{predicted_label}</div>
        <div class="result-label">
            <strong>Output returned by the model:</strong>
            Class {prediction}<br>
            <strong>Meaning:</strong> {interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Class probability")

    st.caption(
        "These percentages are the model's estimated probabilities for "
        "the dataset classes. They are not laboratory measurements or a "
        "guarantee of drinking-water safety."
    )

    for class_label, probability in zip(classes, probabilities):
        percentage = float(probability) * 100
        class_name = LABELS.get(
            class_label,
            f"Class {class_label}",
        )

        color = "#087E8B" if class_label == prediction else "#B7D9DE"

        st.markdown(f"""
        <div class="probability-label">
            <span>{class_name}</span>
            <span>{percentage:.2f}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill"
                 style="width:{percentage:.2f}%;background:{color};">
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Submitted water measurements")

    display_sample = sample.T.reset_index()
    display_sample.columns = ["Parameter", "Value"]
    display_sample["Parameter"] = (
        display_sample["Parameter"]
        .map(PARAMETER_NAMES)
        .fillna(display_sample["Parameter"])
    )

    st.dataframe(
        display_sample,
        use_container_width=True,
        hide_index=True,
    )

    report_text = create_prediction_report(
        sample,
        result,
        timestamp,
    )

    st.markdown("### Download prediction report")

    st.download_button(
        label="Download report (.txt)",
        data=report_text,
        file_name="aqualens_prediction_report.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"download_report_{timestamp}",
    )

    st.caption(
        "The report includes the prediction, class definitions, "
        "probabilities, input measurements, and limitations."
    )


# =====================================================
# 7. OVERVIEW
# =====================================================

page = st.session_state["nav_page"]

if page == "Overview":
    page_header(
        "Water quality, understood.",
        "Analyze water-quality measurements and review machine-learning "
        "model performance.",
    )

    st.markdown("### Model performance")

    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Test accuracy", format_metric(metrics.get("accuracy")))
        c2.metric(
            "Balanced accuracy",
            format_metric(metrics.get("balanced_accuracy")),
        )
        c3.metric("Macro F1-score", format_metric(metrics.get("f1_macro")))
        c4.metric("ROC-AUC", format_metric(metrics.get("roc_auc")))
    else:
        st.info(
            "Evaluation metrics are unavailable. Run your training script "
            "to generate reports/metrics.json."
        )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">Analyze a sample</div>
            <p style="font-size:13px;color:#526B7B;line-height:1.9;">
                Enter nine water-quality measurements, run the classifier,
                review the predicted class, and download the report.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Analyze a sample",
            use_container_width=True,
            on_click=go_to_analyze,
        )

    with right:
        training_rows = (
            metrics.get("training_rows", "N/A") if metrics else "N/A"
        )
        testing_rows = (
            metrics.get("testing_rows", "N/A") if metrics else "N/A"
        )

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">Model details</div>
            <p style="font-size:13px;color:#526B7B;line-height:2;">
                <strong>Algorithm</strong><br>
                Random Forest Classifier<br>
                <strong>Dataset</strong><br>
                Water Potability<br>
                <strong>Training samples</strong><br>
                {training_rows}<br>
                <strong>Testing samples</strong><br>
                {testing_rows}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "AquaLens is a research prototype, not a certified "
        "water-testing instrument."
    )


# =====================================================
# 8. ANALYZE SAMPLE
# =====================================================

elif page == "Analyze Sample":
    page_header(
        "Analyze a water sample",
        "Enter measurements to generate a prediction from the trained model.",
    )

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        st.markdown("### Water parameters")

        with st.form("water_quality_form"):
            a, b = st.columns(2)

            with a:
                ph = st.number_input(
                    "pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=DEFAULT_VALUES["ph"],
                    step=0.1,
                )
                hardness = st.number_input(
                    "Hardness",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Hardness"],
                    step=1.0,
                )
                solids = st.number_input(
                    "Total dissolved solids",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Solids"],
                    step=100.0,
                )
                chloramines = st.number_input(
                    "Chloramines",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Chloramines"],
                    step=0.1,
                )
                sulfate = st.number_input(
                    "Sulfate",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Sulfate"],
                    step=1.0,
                )

            with b:
                conductivity = st.number_input(
                    "Conductivity",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Conductivity"],
                    step=1.0,
                )
                organic_carbon = st.number_input(
                    "Organic carbon",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Organic_carbon"],
                    step=0.1,
                )
                trihalomethanes = st.number_input(
                    "Trihalomethanes",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Trihalomethanes"],
                    step=1.0,
                )
                turbidity = st.number_input(
                    "Turbidity",
                    min_value=0.0,
                    value=DEFAULT_VALUES["Turbidity"],
                    step=0.1,
                )

            submitted = st.form_submit_button(
                "Run water analysis",
                use_container_width=True,
            )

    with result_col:
        st.markdown("### Analysis result")

        if submitted:
            st.session_state["last_sample"] = None
            st.session_state["last_prediction_result"] = None
            st.session_state["last_report_timestamp"] = None

            sample = pd.DataFrame([{
                "ph": ph,
                "Hardness": hardness,
                "Solids": solids,
                "Chloramines": chloramines,
                "Sulfate": sulfate,
                "Conductivity": conductivity,
                "Organic_carbon": organic_carbon,
                "Trihalomethanes": trihalomethanes,
                "Turbidity": turbidity,
            }], columns=FEATURES)

            try:
                prediction, classes, probabilities = predict_sample(sample)

                result = {
                    "prediction": prediction,
                    "classes": classes,
                    "probabilities": [
                        float(probability)
                        for probability in probabilities
                    ],
                }

                timestamp = datetime.now().astimezone().strftime(
                    "%Y-%m-%d_%H-%M-%S_%Z"
                )

                st.session_state["last_sample"] = sample.copy()
                st.session_state["last_prediction_result"] = result
                st.session_state["last_report_timestamp"] = timestamp

                history_item = {
                    "Timestamp": datetime.now().astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    ),
                    **sample.iloc[0].to_dict(),
                    "Predicted class": prediction,
                    "Predicted label": LABELS.get(
                        prediction,
                        str(prediction),
                    ),
                }

                for class_label, probability in zip(classes, probabilities):
                    history_item[
                        f"Class {class_label} probability"
                    ] = round(float(probability), 6)

                st.session_state["prediction_history"].append(history_item)

            except (ValueError, TypeError) as error:
                st.error(str(error))

            except Exception:
                logger.exception("Prediction failed.")
                st.error(
                    "Analysis failed. Check the input measurements "
                    "and saved model."
                )

        sample_to_show = st.session_state.get("last_sample")
        result_to_show = st.session_state.get("last_prediction_result")
        timestamp_to_show = st.session_state.get("last_report_timestamp")

        if (
            sample_to_show is not None
            and result_to_show is not None
            and timestamp_to_show is not None
        ):
            show_prediction(
                sample_to_show,
                result_to_show,
                timestamp_to_show,
            )
        else:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">No prediction yet</div>
                <p style="font-size:13px;color:#8293A0;">
                    Submit the measurements to view the result and generate
                    a downloadable report.
                </p>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 9. MODEL REPORTS
# =====================================================

elif page == "Model Reports":
    page_header(
        "Model evaluation",
        "Performance metrics and classification results from the held-out "
        "test dataset.",
    )

    if metrics is not None:
        metric_descriptions = {
            "accuracy": (
                "Accuracy",
                "Proportion of test samples classified correctly.",
            ),
            "balanced_accuracy": (
                "Balanced accuracy",
                "Average recall across classes.",
            ),
            "precision_macro": (
                "Macro precision",
                "Average precision across classes, giving each class equal weight.",
            ),
            "recall_macro": (
                "Macro recall",
                "Average recall across classes, giving each class equal weight.",
            ),
            "f1_macro": (
                "Macro F1-score",
                "Harmonic mean of macro precision and macro recall.",
            ),
            "roc_auc": (
                "ROC-AUC",
                "Measures how well the model ranks samples from the two classes.",
            ),
            "cv_balanced_accuracy": (
                "Cross-validation balanced accuracy",
                "Average balanced accuracy across validation folds, if recorded.",
            ),
            "best_cv_balanced_accuracy": (
                "Best CV balanced accuracy",
                "Best cross-validation balanced accuracy, if recorded.",
            ),
        }

        rows = []

        for key, (name, description) in metric_descriptions.items():
            if key in metrics:
                rows.append({
                    "Metric": name,
                    "Score": format_metric(metrics[key]),
                    "Meaning": description,
                })

        st.markdown("### Evaluation metrics")

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No recognized evaluation metrics were found.")

        c1, c2, c3 = st.columns(3)

        c1.metric("Training samples", metrics.get("training_rows", "N/A"))
        c2.metric("Testing samples", metrics.get("testing_rows", "N/A"))
        c3.metric("Total samples", metrics.get("dataset_rows_used", "N/A"))

        if "best_params" in metrics:
            st.markdown("### Model parameters")

            params = metrics["best_params"]

            if isinstance(params, dict):
                params_df = pd.DataFrame(
                    [
                        {"Parameter": key, "Value": str(value)}
                        for key, value in params.items()
                    ]
                )
                st.dataframe(
                    params_df,
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("View raw metrics JSON"):
            st.json(metrics)

    else:
        st.info(
            "The metrics file is not available. Run your training script "
            "to generate the evaluation reports."
        )

    st.markdown("---")
    st.markdown("### Confusion matrix")

    confusion_path = REPORTS_DIR / "confusion_matrix.png"

    if confusion_path.exists():
        st.image(str(confusion_path), use_container_width=True)
    else:
        st.caption("Confusion matrix image has not been generated.")

    st.markdown("### Classification report")

    report_path = REPORTS_DIR / "classification_report.txt"

    if report_path.exists():
        st.code(
            report_path.read_text(encoding="utf-8"),
            language="text",
        )
    else:
        st.caption("Classification report has not been generated.")

    st.markdown("### Feature importance")

    importance_path = REPORTS_DIR / "feature_importance.png"
    importance_csv = REPORTS_DIR / "feature_importance.csv"

    if importance_path.exists():
        st.image(str(importance_path), use_container_width=True)

    if importance_csv.exists():
        try:
            importance_df = pd.read_csv(importance_csv)
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True,
            )
        except (OSError, ValueError, pd.errors.ParserError):
            st.caption("Feature importance data could not be read.")


# =====================================================
# 10. PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":
    page_header(
        "Prediction history",
        "Review and export samples analyzed during this session.",
    )

    history = st.session_state["prediction_history"]

    if history:
        history_df = pd.DataFrame(history)

        st.metric("Samples analyzed", len(history))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="aqualens_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("Clear history", use_container_width=True):
            st.session_state["prediction_history"] = []
            st.rerun()

    else:
        st.info("No predictions recorded in this session yet.")

    st.caption(
        "History is temporary and may be cleared when the Streamlit session resets."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    AQUALENS | WATER INTELLIGENCE<br>
    Machine-learning research prototype
</div>
""", unsafe_allow_html=True)
