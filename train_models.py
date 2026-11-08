"""
Trains 5 classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)
on a dataset you specify, to predict a target column you specify.

Downloads the dataset if given a URL (caching it locally), splits it into
train/test, saves the test split as test_data.csv, trains each model as a
full preprocessing+classifier pipeline, saves each to model/*.joblib, and
writes the evaluation metrics comparison table to model/metrics.csv.

The script has no knowledge of any particular dataset's columns baked in —
categorical vs numeric features are auto-detected by dtype, and the target
column, dropped columns, and data source are all given via CLI arguments.

Usage:
    python train_models.py --data path/or/url/to/data.csv --target "Target Column"
    python train_models.py --data path/or/url/to/data.zip --target y --drop id,notes
"""

import argparse
import io
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASSIFIER_BUILDERS = {
    "logistic_regression": lambda random_state: LogisticRegression(max_iter=1000),
    "decision_tree": lambda random_state: DecisionTreeClassifier(
        random_state=random_state
    ),
    "knn": lambda random_state: KNeighborsClassifier(),
    "naive_bayes": lambda random_state: GaussianNB(),
    "svm": lambda random_state: SVC(probability=True, random_state=random_state),
}


ACRONYMS = {"knn", "svm"}


def display_name(model_key):
    return " ".join(
        word.upper() if word in ACRONYMS else word.capitalize()
        for word in model_key.split("_")
    )


def resolve_dataset_path(data_arg, cache_dir):
    """Returns a local CSV path for `data_arg`, downloading/extracting if needed."""
    if os.path.exists(data_arg):
        return data_arg

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading dataset from {data_arg} ...")
    with urllib.request.urlopen(data_arg) as response:
        content = response.read()

    if data_arg.lower().endswith(".zip") or content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            cached_path = os.path.join(cache_dir, os.path.basename(csv_name))
            with zf.open(csv_name) as f:
                with open(cached_path, "wb") as out:
                    out.write(f.read())
    else:
        cached_path = os.path.join(cache_dir, os.path.basename(data_arg))
        with open(cached_path, "wb") as out:
            out.write(content)

    print(f"Cached dataset at {cached_path}")
    return cached_path


def load_data(csv_path, target_col, drop_cols):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X = df[feature_cols]
    y = df[target_col]
    return X, y, categorical_features, numeric_features


def build_models(categorical_features, numeric_features, random_state):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    return {
        name: Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", builder(random_state))]
        )
        for name, builder in CLASSIFIER_BUILDERS.items()
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    classes = model.classes_
    multiclass = len(classes) > 2
    average = "weighted" if multiclass else "binary"

    y_proba = model.predict_proba(X_test)
    if multiclass:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    else:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Local path or URL to a CSV/zip dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument(
        "--drop", default="", help="Comma-separated columns to exclude from features"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--test-data-out", default="test_data.csv")
    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop.split(",") if c.strip()]

    csv_path = resolve_dataset_path(args.data, args.cache_dir)
    X, y, categorical_features, numeric_features = load_data(
        csv_path, args.target, drop_cols
    )
    print(f"Categorical features: {categorical_features}")
    print(f"Numeric features: {numeric_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    test_df = X_test.copy()
    test_df[args.target] = y_test
    test_df.to_csv(args.test_data_out, index=False)
    print(f"Saved test split ({len(test_df)} rows) to {args.test_data_out}")

    os.makedirs(args.model_dir, exist_ok=True)
    models = build_models(categorical_features, numeric_features, args.random_state)

    rows = []
    for name, pipeline in models.items():
        print(f"Training {display_name(name)} ...")
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        rows.append({"Model": display_name(name), **metrics})

        model_path = os.path.join(args.model_dir, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"  saved -> {model_path}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.model_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== Evaluation metrics (test split) ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved comparison table -> {metrics_path}")


if __name__ == "__main__":
    main()
