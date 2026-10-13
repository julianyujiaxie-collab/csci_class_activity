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

    # Figure 1

    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 120,
    })

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))

    bcl2_groups = [
        X_train[y_train == label, 0]
        for label in [0, 1]
    ]

    boxplot = ax1.boxplot(
        bcl2_groups,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for box, color in zip(boxplot["boxes"], CLASS_COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.65)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_xlabel("ER status measured by IHC")
    ax1.set_ylabel("BCL2 expression (z-score)")
    ax1.set_title("Figure 1: BCL2 expression by ER status\nTraining set")
    ax1.grid(axis="y", alpha=0.25)

    fig1.tight_layout()

    # Figure 2

    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 120,
    })

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))

    gata3_groups = [
        X_train[y_train == label, 1]
        for label in [0, 1]
    ]

    boxplot = ax2.boxplot(
        gata3_groups,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for box, color in zip(boxplot["boxes"], CLASS_COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.65)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_xlabel("ER status measured by IHC")
    ax2.set_ylabel("GATA3 expression (z-score)")
    ax2.set_title("Figure 2: GATA3 expression by ER status\nTraining set")
    ax2.grid(axis="y", alpha=0.25)

    fig2.tight_layout()

    # Figure 3

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))

    for label, class_name, color in zip(
        [0, 1], CLASS_NAMES, CLASS_COLORS
    ):
        mask = y_train == label

        ax3.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            label=class_name,
            color=color,
            alpha=0.45,
            s=20,
            edgecolors="none",
        )

    ax3.set_xlabel("BCL2 expression (z-score)")
    ax3.set_ylabel("GATA3 expression (z-score)")
    ax3.set_title("Figure 3: BCL2 and GATA3 expression\nTraining set")
    ax3.legend(title="IHC label")
    ax3.grid(alpha=0.2)

    fig3.tight_layout()

    # Figure 4: Test-set confusion matrix

    fig4, ax4 = plt.subplots(figsize=(6, 4.5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        model_pred,
        labels=[0, 1],
        display_labels=CLASS_NAMES,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax4,
    )

    ax4.set_title(
        "Figure 4: Logistic regression — test set"
    )

    fig4.tight_layout()
    
    figures = [
            (fig1, "figure_1_bcl2_boxplot.png"),
            (fig2, "figure_2_gata3_boxplot.png"),
            (fig3, "figure_3_gene_expression_scatter.png"),
            (fig4, "figure_4_confusion_matrix.png"),
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
