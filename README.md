# Breast Cancer Gene Expression Analysis

A small classification project using the [METABRIC dataset](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) to predict estrogen receptor (ER) status measured by immunohistochemistry (IHC) from **BCL2** and **GATA3** expression z-scores.

## Hypothesis

A logistic regression model using BCL2 and GATA3 expression can distinguish ER-positive from ER-negative tumors better than a majority-class baseline on held-out samples.

## Method

- Remove samples with missing selected features or labels and encode ER status.
- Use a stratified 80/20 train/test split with `random_state=0`.
- Fit a training-set standardization pipeline and logistic regression with balanced class weights.
- Compare against a majority-class baseline using accuracy, balanced accuracy, and ROC-AUC on the test set.

The label mapping handles the dataset's original spelling of positive ER status: `Positve`.

## Run

From this directory, install the dependencies and create the data and output directories:

```bash
mkdir -p dataset figures
```

Download `METABRIC_RNA_Mutation.csv` from Kaggle and place it in `dataset/`, then run:

```bash
python3 analysis.py
```

## Output

The script prints test-set metrics and saves four figures in `figures/`:

1. BCL2 expression by ER status — training set.
2. GATA3 expression by ER status — training set.
3. BCL2 versus GATA3 expression — training set.
4. Logistic regression confusion matrix — test set.

Results describe predictive associations in one train/test split; they do not establish causality.
