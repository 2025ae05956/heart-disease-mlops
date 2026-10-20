import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
RANDOM_STATE = 42


def clean_raw(path):
    """Raw UCI file -> clean dataframe with binary target (same steps as Task 1)."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor():
    """Scaling for numbers, passthrough for 0/1 columns, one-hot for categories."""
    return ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC),
        ("keep", "passthrough", BINARY),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def build_final_model():
    """The selected model from Task 2 with its tuned settings."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(C=1, class_weight="balanced", max_iter=1000)),
    ])
