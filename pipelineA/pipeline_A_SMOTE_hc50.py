""""
Pipeline A SMOTE
SMOTE oversamples the minority classes of the training set by generating synthetic points between real minority-class neighbors.



"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # so it saves the plots
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm

# setting up a config so we can modify it easily later
RANDOM_STATE = 42
C_VALUE = 1.0
GAMMA = "scale"
KERNEL = "rbf"

# non features columns
META_COLUMNS = {"SequenceIndex", "Sequence", "Class", "Split", "log10mic", "log10hc50"}
# name of the classes
CLASS_NAMES = ["Selective", "Pure Hemolytic", "Both Active/Toxic", "Inactive"]

#INPUT_FILE = "lasso_pipeline_A_HC50_features.csv"
INPUT_FILE = "lasso_pipeline_A_HC50_features.csv"

# load the LASSO-selected feature matrix, split it via the split column
def load_xy(path):
    df=pd.read_csv(path)
    if "Split" not in df.columns:
        raise ValueError("error loading the split column")
    feature_cols = [c for c in df.columns if c not in META_COLUMNS] # build the list, go through every column and keep it if it's one of the columns in meta_columns
    train = df["Split"] == "train"
    test = df["Split"] == "test"
    print("debug 1 discard")
    class_codes = df["Class"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    X_train = df.loc[train, feature_cols].to_numpy()
    y_train = class_codes[train].to_numpy()
    X_test = df.loc[test, feature_cols].to_numpy()
    y_test = class_codes[test].to_numpy()
    #end of debug
    return X_train, y_train, X_test, y_test, feature_cols

def counts(y):
    c, n = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(c, n)}

def main():
    path = INPUT_FILE
    X_train, y_train, X_test, y_test, feature_cols = load_xy(path)
    print(f"Features column length: {len(feature_cols)}.")
    print(f"Train length: {len(y_train)}.")
    print(f"Test length: {len(y_test)}.")

    # deciding k_neighbors
    min_count = min(counts(y_train).values())
    k_neighbors = min(5, max(1, min_count-1))

    #resampling the train set, not using the class_weight
    smote = SMOTE(random_state = RANDOM_STATE, k_neighbors = k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"train class count after SMOTE: {counts(y_res)}")

    #svc
    svm = sklearn.svm.SVC(kernel=KERNEL, C=C_VALUE, gamma=GAMMA, random_state=RANDOM_STATE)
    svm.fit(X_res,y_res) # the learning set the SVM looks at

    #test/making predictions
    y_pred_test = svm.predict(X_test)
    y_pred_train = svm.predict(X_train) #original pre-smote data
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train)
    test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    macro_f1 = sklearn.metrics.f1_score(y_test, y_pred_test, average="macro")
    weighted_f1 = sklearn.metrics.f1_score(y_test, y_pred_test, average="weighted")

    #displaying the results
    print(f"Training accuracy: {train_acc:.3f}.")
    print(f"Test accuracy: {test_acc:.3f}.")
    print()
    print(f"Macro f1 score: {macro_f1:.3f}.")
    print(f"Weighted f1 score: {weighted_f1:.3f}.")
    print("classification_report")
    print(sklearn.metrics.classification_report(
        y_test, y_pred_test,
        labels=sorted(counts(y_test)),
        target_names=[CLASS_NAMES[i] for i in sorted(counts(y_test))],
        zero_division=0,
    ))

    #cross-validation
    cv_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ("svm", sklearn.svm.SVC(kernel=KERNEL, C=C_VALUE, gamma=GAMMA,
                                 random_state=RANDOM_STATE)),
    ])
    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    macro_cv = sklearn.model_selection.cross_val_score(
        cv_pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    weighted_cv = sklearn.model_selection.cross_val_score(
        cv_pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"CV macro    F1: {macro_cv.mean():.3f} +/- {macro_cv.std():.3f}")
    print(f"CV weighted F1: {weighted_cv.mean():.3f} +/- {weighted_cv.std():.3f}")

    # plot (copy-pasted from pip A)
    labels = sorted(set(y_test) | set(y_pred_test))
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred_test, labels=labels)
    disp = sklearn.metrics.ConfusionMatrixDisplay(
        cm, display_labels=[CLASS_NAMES[i] for i in labels])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="RdPu",
              values_format="d")
    disp.im_.colorbar.set_label("Number of peptides", fontsize=13)
    #ax.set_title("Pipeline C - HC50 (SMOTE)", fontsize=14)
    ax.set_title("Pipeline A - HC50 (SMOTE)", fontsize=14)
    ax.set_xlabel("Predicted Class", fontsize=14)
    ax.set_ylabel("True Class", fontsize=14)
    fig.tight_layout()
    #out = "pipeline_C_confusion_hc50.png"
    out = "pipeline_A_SMOTE_confusion_hc50.png"
    fig.savefig(out, dpi=150)

if __name__ == "__main__":
    main()
