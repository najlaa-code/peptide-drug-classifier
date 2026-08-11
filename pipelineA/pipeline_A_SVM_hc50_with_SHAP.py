"""
SVM for pipeline A (HC50) along with SHAP
Train an SVM (RBF, class_weight="balanced") on the LASSO selected HC50 features from pipeline A. Uses the same 
train/test split. Reports classification metrictions + 5 fold CV + saves a confusion matrix plot.

Trains a 2nd SVM (same parameters, probability=True due to KernelExplainer needing predict_proba). Runs SHAP on 
it to see which descriptors are actually driving the predictions. Special emphasis on the minority against the 
majority class.

Input:
- lasso_pipeline_A_HC50_features.csv

Output:
- prints to the console: train/test sizes, feature count, classification report, mean, +/- std of 5-fold CV 
weighted F1, top-15 descriptors for the rare class, top-15 for the majority class, shared descriptor count 
+ Jaccard overlap
- svm_confusion_matrix_pipeline_A_HC50.png
- shap_pipeline_A_hc50_mean_abs.csv
-shap_bar_pipeline_A_HC50_<class>.png (one bar plot per class)
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np
import shap

BASE_DIR = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS"
INPUT_CSV = os.path.join(BASE_DIR, "lasso_pipeline_A_HC50_features.csv")
print("HC50 results:")
OUT_CM    = os.path.join(BASE_DIR, "svm_confusion_matrix_pipeline_A_HC50.png")

df = pd.read_csv(INPUT_CSV)
LABEL_COL = "Class"

META_DROP = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic", "Split"]
META_DROP = [c for c in META_DROP if c in df.columns]
train_mask = df["Split"] == "train" # using the same test/train splits from pipeline A
test_mask = df["Split"] == "test"
X_train = df.loc[train_mask].drop(columns=META_DROP)
y_train = df.loc[train_mask, LABEL_COL]
X_test  = df.loc[test_mask].drop(columns=META_DROP)
y_test  = df.loc[test_mask, LABEL_COL]
print(f"Training size: {len(X_train)} rows. Testing size: {len(X_test)} rows.")
print(f"Features: {X_train.shape[1]}")

# Training the SVM
svm = SVC(
    kernel="rbf", # try a diff one
    C=1.0, # fine tune
    gamma="scale",
    class_weight="balanced",
    random_state=42
)
svm.fit(X_train, y_train)

#Evaluating the SVC
y_pred = svm.predict(X_test)
print("Classification results:")
print(classification_report(y_test, y_pred))
# cv --> cross validation on train only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="f1_weighted")
print(f"Mean of cross-validation scores: {cv_scores.mean():.3f}")
print(f"Standard deviation of cross-validation scores: {cv_scores.std():.3f}")

#confusion matrix for FP, FN, TP and TN
# cm stands for confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
# plot

cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="RdPu", values_format="d")
disp.im_.colorbar.set_label("Number of peptides", fontsize=13)

ax.set_xlabel("Predicted Class", fontsize=14)
ax.set_ylabel("True Class", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.set_title("SVM Confusion Matrix Pipeline A (HC50 features)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_CM, dpi=150)
plt.close()
svm_probability = SVC(
    kernel = "rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
svm_probability.fit(X_train, y_train)
K_BACKGROUND = 25
N_PER_CLASS  = 30
TOPN         = 15

feature_names = list(X_train.columns)
classes = list(svm_probability.classes_)

np.random.seed(42)
background = shap.kmeans(X_train.values, K_BACKGROUND)
print("debug 1: building explainer")
explainer = shap.KernelExplainer(svm_probability.predict_proba, background)
# picking which molecules to explain, so looping through each class, taking n (n_per_class) test molecules randomly
rng = np.random.RandomState(42)
explain_idx = []
for c in classes:
    idx_c = y_test.index[y_test == c].to_numpy()
    take  = min(N_PER_CLASS, len(idx_c))
    explain_idx += list(rng.choice(idx_c, size=take, replace=False))
X_explain = X_test.loc[explain_idx]

# shap happens here
shap_values = explainer.shap_values(X_explain.values)
if isinstance(shap_values, list):
    sv_list = shap_values
else:
    sv_list = [shap_values[:, :, i] for i in range(len(classes))]

# making the plots and tables
imp = pd.DataFrame(
    {c: pd.Series(np.abs(sv_list[i]).mean(axis=0), index=feature_names)
     for i, c in enumerate(classes)}
)
imp.to_csv(os.path.join(BASE_DIR, "shap_pipeline_A_hc50_mean_abs.csv"))


def safe(s):
    return s.replace(" ", "_").replace(":", "").replace("/", "-")


for i, c in enumerate(classes):
    plt.figure()
    shap.summary_plot(sv_list[i], X_explain.values, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=TOPN)
    plt.title(f"SHAP mean |value| — {c} (HC50, Pipeline A)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"shap_bar_pipeline_A_HC50_{safe(c)}.png"), dpi=150)
    plt.close()

RARE = next(c for c in classes if "Both Active" in c)
MAJ = next(c for c in classes if "Inactive" in c)
rare_top = list(imp[RARE].sort_values(ascending=False).head(TOPN).index)
maj_top = list(imp[MAJ].sort_values(ascending=False).head(TOPN).index)
overlap = set(rare_top) & set(maj_top)
jacc = len(overlap) / len(set(rare_top) | set(maj_top))

print("\nDiagnostic report:")
print(f"Top {TOPN} descriptors for RARE class ({RARE}):")
print("  " + ", ".join(rare_top))
print(f"\nTop {TOPN} descriptors for MAJORITY class ({MAJ}):")
print("  " + ", ".join(maj_top))
print(f"\nShared descriptors: {len(overlap)}  |  Jaccard overlap: {jacc:.2f}")
print("  " + (", ".join(sorted(overlap)) if overlap else "(none)"))

# if high overlap, model uses majority-class descriptors for everything, the rare class has no distinctive signal (i.e imbalance artifact)
# if low overlap, rare class has its own descriptor signal but is being out-voted (signal exists, not enough samples)
