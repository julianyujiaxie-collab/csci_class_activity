from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

CSV_PATH = Path("dataset/METABRIC_RNA_Mutation.csv")

FEATURES = ["bcl2", "gata3"]
TARGET = "er_status_measured_by_ihc"

TEST_SIZE = 0.2
RANDOM_STATE = 0

OUTPUT_DIR = Path("figures")

CLASS_NAME = ["ER-", "ER+"]

def main():

    # Load the selected columns

    df = pd.read_csv(CSV_PATH)
    
    required_cols = FEATURES + [TARGET]
    data = df[required_cols].copy()

    # Convert expression values to numbers.
    # Invalid values become NaN and are removed below.
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # Normalize labels and encode Negative = 0, Positive = 1.
    label_text = (
        data[TARGET]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    data["label"] = label_text.map({
        "negative": 0,
        "positive": 1,
    })

    data = data.dropna(subset=FEATURES + ["label"]).copy()

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)

    # Split train/test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Fit the baseline and logistic regression

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_train, y_train)
    
    # Evaluate on the test set

    baseline_pred = baseline.predict(X_test)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]

    model_pred = model.predict(X_test)

    # ROC-AUC uses positive-class probabilities, not hard labels.
    model_prob = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame(
        {
            "Accuracy": [
                accuracy_score(y_test, baseline_pred),
                accuracy_score(y_test, model_pred),
            ],
            "Balanced accuracy": [
                balanced_accuracy_score(y_test, baseline_pred),
                balanced_accuracy_score(y_test, model_pred),
            ],
            "ROC-AUC": [
                roc_auc_score(y_test, baseline_prob),
                roc_auc_score(y_test, model_prob),
            ],
        },
        index=["Majority-class baseline", "Logistic regression"],
    )

    print("\nTest-set results:")
    print(results.round(3).to_string())

    
if __name__ == '__main__':
    main()
