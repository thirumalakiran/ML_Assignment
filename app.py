"""
Streamlit app for comparing 5 classic classification algorithms.

No training happens in the app. All 5 models are loaded from model/
(already trained offline by train_models.py); once a test CSV is
uploaded, each saved model is used only to .predict() on it so the
comparison table, observations, and per-model confusion matrix /
classification report can be computed.

No dataset is hardcoded: column names and the number of classes are
discovered from whatever CSV the user uploads.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"
SAMPLE_TEST_DATA_PATH = "test_data.csv"

st.set_page_config(page_title="ML Classifier Comparison App", layout="wide")


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


ACRONYMS = {"knn", "svm"}


def display_name(model_key):
    return " ".join(
        word.upper() if word in ACRONYMS else word.capitalize()
        for word in model_key.split("_")
    )


def list_saved_models():
    if not os.path.isdir(MODEL_DIR):
        return {}
    files = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib"))
    return {display_name(os.path.splitext(f)[0]): f for f in files}


def build_observation(row):
    acc, auc, prec, rec, f1, mcc = (
        row["Accuracy"],
        row["AUC"],
        row["Precision"],
        row["Recall"],
        row["F1"],
        row["MCC"],
    )
    if rec < 0.3:
        return (
            f"High accuracy ({acc:.1%}) but low recall ({rec:.1%}, MCC {mcc:.2f}) — "
            f"this usually means the positive/minority class is rare and the model "
            f"defaults to predicting the majority class most of the time."
        )
    if mcc >= 0.5:
        return (
            f"Strong, balanced performance (F1 {f1:.1%}, MCC {mcc:.2f}) — correctly "
            f"identifies most classes without sacrificing precision."
        )
    return (
        f"Moderate performance (F1 {f1:.1%}, MCC {mcc:.2f}, AUC {auc:.1%}) — better "
        f"than guessing but still misses a meaningful share of cases."
    )


def compute_metrics(y_true, y_pred, y_proba, classes):
    multiclass = len(classes) > 2
    average = "weighted" if multiclass else "binary"

    pos_label = classes[1] if not multiclass else None
    kwargs = {} if multiclass else {"pos_label": pos_label}

    if multiclass:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=classes
        )
    else:
        pos_index = list(classes).index(pos_label)
        auc = roc_auc_score(y_true, y_proba[:, pos_index])

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_true, y_pred, average=average, zero_division=0, **kwargs
        ),
        "Recall": recall_score(
            y_true, y_pred, average=average, zero_division=0, **kwargs
        ),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0, **kwargs),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_saved_model(filename, X, y_true):
    """Loads an already-trained model and predicts on X — never fits/retrains."""
    model = load_model(filename)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = compute_metrics(y_true, y_pred, y_proba, model.classes_)
    return model, y_pred, metrics


st.title("ML Classifier Comparison App")
st.caption(
    "Compare 5 pretrained classic classification algorithms (Logistic "
    "Regression, Decision Tree, KNN, Naive Bayes, SVM) on your own test data. "
    "The app only runs inference with already-trained models — it never "
    "trains or retrains a model on the data you upload."
)

saved_models = list_saved_models()

if not saved_models:
    st.error(
        "No trained models found. Run `python train_models.py --data <path/url> "
        "--target <target_column>` first to train and save the 5 models."
    )
    st.stop()

st.header("1. Upload Test Data")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload a test CSV (must include the feature columns and the target column)",
        type="csv",
    )
with col2:
    if os.path.exists(SAMPLE_TEST_DATA_PATH):
        with open(SAMPLE_TEST_DATA_PATH, "rb") as f:
            st.download_button(
                "Download sample test data",
                data=f.read(),
                file_name="test_data.csv",
                mime="text/csv",
            )

if uploaded_file is None:
    st.info(
        "Upload a CSV above (or download the sample) to see the model "
        "comparison table, observations, and per-model evaluation."
    )
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

columns = list(data.columns)
col_a, col_b = st.columns([1, 2])
with col_a:
    target_col = st.selectbox("Target column", columns, index=len(columns) - 1)
with col_b:
    feature_cols = st.multiselect(
        "Feature columns",
        [c for c in columns if c != target_col],
        default=[c for c in columns if c != target_col],
    )

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

X = data[feature_cols]
y_true = data[target_col]

results = {}
errors = {}
for model_name, filename in saved_models.items():
    try:
        model, y_pred, metrics = evaluate_saved_model(filename, X, y_true)
        results[model_name] = {"model": model, "y_pred": y_pred, "metrics": metrics}
    except Exception as e:
        errors[model_name] = str(e)

if errors:
    st.warning(
        "Some models could not run on the selected columns (likely a schema "
        "mismatch with what they were trained on): "
        + "; ".join(f"{name} — {msg}" for name, msg in errors.items())
    )

if not results:
    st.error(
        "None of the saved models could run on the selected columns. Make sure "
        "the feature columns match the schema the models were trained on."
    )
    st.stop()

metrics_df = pd.DataFrame(
    [{"Model": name, **r["metrics"]} for name, r in results.items()]
)

st.header("2. Model Comparison Table")
st.caption("Metrics computed by running each saved model on the uploaded data.")
st.dataframe(
    metrics_df.set_index("Model").style.format("{:.4f}"),
    use_container_width=True,
)

st.header("3. Observations")
obs_rows = [
    {"Model": row["Model"], "Observation": build_observation(row)}
    for _, row in metrics_df.iterrows()
]
st.table(pd.DataFrame(obs_rows).set_index("Model"))

winner_row = metrics_df.loc[metrics_df["MCC"].idxmax()]
st.success(
    f"**Overall winner: {winner_row['Model']}** — highest Matthews Correlation "
    f"Coefficient (MCC = {winner_row['MCC']:.3f}), the most reliable single metric "
    f"when classes are imbalanced, since it accounts for all four confusion-matrix "
    f"cells rather than being dominated by the majority class the way plain "
    f"accuracy can be."
)

st.header("4. Inspect a Model")
selected_model_name = st.selectbox("Model", list(results.keys()))
selected = results[selected_model_name]
y_pred = selected["y_pred"]
result_metrics = selected["metrics"]

st.subheader(f"Results — {selected_model_name}")
metric_cols = st.columns(6)
for col, (name, value) in zip(metric_cols, result_metrics.items()):
    col.metric(name, f"{value:.4f}")

labels = sorted(set(y_true.unique()) | set(y_pred))
label_strs = [str(l) for l in labels]

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_strs,
    yticklabels=label_strs,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(
    y_true, y_pred, labels=labels, target_names=label_strs, output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))
