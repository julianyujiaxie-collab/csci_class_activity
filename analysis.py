from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
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

CLASS_NAMES = ["ER-", "ER+"]
CLASS_COLORS = ["#377eb8", "#e66101"]

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
        "positve": 1,
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
        },
        index=["Majority-class baseline", "Logistic regression"],
    )

    print("\nTest-set results:")
    print(results.round(3).to_string())

    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 120,
    })

    # Figure 1

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))

    for label, class_name, color in zip(
        [0, 1], CLASS_NAMES, CLASS_COLORS
    ):
        mask = y_train == label

        ax1.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            label=class_name,
            color=color,
            alpha=0.45,
            s=20,
            edgecolors="none",
        )

    ax1.set_xlabel("BCL2 expression (z-score)")
    ax1.set_ylabel("GATA3 expression (z-score)")
    ax1.set_title("Figure 1: BCL2 and GATA3 expression")
    ax1.legend(title="IHC label")
    ax1.grid(alpha=0.2)

    fig1.tight_layout()

    # Figure 2: Learning curve using only the training set.

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=cv,
        scoring="balanced_accuracy",
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))

    ax2.plot(
        train_sizes, train_mean, "o-",
        color=CLASS_COLORS[0], label="Training",
    )
    ax2.plot(
        train_sizes, validation_mean, "o-",
        color=CLASS_COLORS[1], label="Cross-validation",
    )

    ax2.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        color=CLASS_COLORS[0], alpha=0.15,
    )
    ax2.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        color=CLASS_COLORS[1], alpha=0.15,
    )
    ax2.axhline(
        0.5, color="gray", linestyle="--", linewidth=1,
        label="Majority-class baseline",
    )

    ax2.set_xlabel("Number of training samples per fit")
    ax2.set_ylabel("Balanced accuracy")
    ax2.set_title("Figure 2: Logistic regression learning curve")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig2.tight_layout()

    # Figure 3: Test-set confusion matrix

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        model_pred,
        labels=[0, 1],
        display_labels=CLASS_NAMES,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax3,
    )

    ax3.set_title(
        "Figure 3: Logistic regression — test set"
    )

    fig3.tight_layout()
    
    figures = [
            (fig1, "figure_1_gene_expression_scatter.png"),
            (fig2, "figure_2_learning_curve.png"),
            (fig3, "figure_3_confusion_matrix.png"),
        ]

    for figure, filename in figures:
        figure.savefig(
            OUTPUT_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )

    print(f"\nFigures saved to: {OUTPUT_DIR.resolve()}")

if __name__ == '__main__':
    main()
