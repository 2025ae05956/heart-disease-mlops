"""Train the final heart disease model, evaluate it, log to MLflow and save outputs.

Usage: python src/train.py --min-accuracy 0.80
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from heart_pipeline import FEATURES, RANDOM_STATE, build_final_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_clean.csv"
OUT = ROOT / "outputs"
TRUSTED = ["sklearn.compose._column_transformer._RemainderColsList"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.80,
                        help="fail the run if test accuracy is below this value")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    # 1. data
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Loaded {len(df)} rows -> train {len(X_train)}, test {len(X_test)}")

    # 2. cross-validation on the training data
    model = build_final_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]
    cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    metrics = {f"cv_{m}": round(float(cv_scores[f"test_{m}"].mean()), 4) for m in scoring}

    # 3. final fit + test-set evaluation
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics.update({
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, prob), 4),
    })
    print(json.dumps(metrics, indent=2))

    # 4. save model, metrics and plots
    joblib.dump(model, OUT / "heart_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[0], colorbar=False)
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax[1])
    ax[0].set_title("Confusion matrix (test)")
