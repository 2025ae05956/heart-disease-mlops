"""Unit tests for data processing (clean_raw) and model code (pipeline)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_pipeline import (FEATURES, NUMERIC, RANDOM_STATE, build_final_model,  # noqa: E402
                            build_preprocessor, clean_raw)

DATA = ROOT / "data" / "heart_clean.csv"

# small sample in the raw UCI format: '?' = missing, last column num = 0..4, last row duplicated
RAW_ROWS = [
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0",
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2",
    "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1",
    "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0",
    "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0",
    "56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4",
    "56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4",
]


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text("\n".join(RAW_ROWS) + "\n")
    return path


@pytest.fixture
def data():
    df = pd.read_csv(DATA)
    return df[FEATURES], df["target"]


# ---------------- data processing tests ----------------
def test_clean_raw_creates_binary_target(raw_file):
    df = clean_raw(raw_file)
    assert "num" not in df.columns
    assert df["target"].tolist() == [0, 1, 1, 0, 0, 1]   # num 1-4 -> 1, num 0 -> 0


def test_clean_raw_fills_missing_values(raw_file):
    assert clean_raw(raw_file).isna().sum().sum() == 0


def test_clean_raw_removes_duplicates(raw_file):
    assert len(clean_raw(raw_file)) == 6


def test_clean_raw_keeps_all_features(raw_file):
    assert set(FEATURES) <= set(clean_raw(raw_file).columns)


def test_clean_dataset_is_valid():
    df = pd.read_csv(DATA)
    assert len(df) > 250
    assert df.isna().sum().sum() == 0
    assert set(df["target"].unique()) == {0, 1}


# ---------------- model / pipeline tests ----------------
def test_preprocessor_output_has_no_missing_values(data):
    X, _ = data
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any()


def test_numeric_features_are_scaled(data):
    X, _ = data
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert np.allclose(out[:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-6)


def test_model_returns_valid_probabilities(data):
    X, y = data
    prob = build_final_model().fit(X, y).predict_proba(X)
    assert prob.shape == (len(X), 2)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_model_accuracy_above_baseline(data):
    X, y = data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    assert build_final_model().fit(X_tr, y_tr).score(X_te, y_te) > 0.75


def test_training_is_reproducible(data):
    X, y = data
    p1 = build_final_model().fit(X, y).predict_proba(X)
    p2 = build_final_model().fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)


def test_unseen_category_does_not_crash(data):
    X, y = data
    model = build_final_model().fit(X, y)
    patient = X.head(1).copy()
    patient["thal"] = 5            # value never seen in training
    assert model.predict(patient)[0] in (0, 1)


def test_saved_model_gives_same_predictions(data, tmp_path):
    X, y = data
    model = build_final_model().fit(X, y)
    joblib.dump(model, tmp_path / "m.joblib")
    assert np.allclose(joblib.load(tmp_path / "m.joblib").predict_proba(X), model.predict_proba(X))
