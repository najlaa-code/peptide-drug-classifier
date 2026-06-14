"""
SVM for pipeline A
MORE HERE
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

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
print(f"Training size: {len(X_train)} rows. Testing size: {test_mask, LABEL_COL} rows.")
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
print(f"Standard deviation fo cross-validation scores: {cv_scores.std():.3f}")

#confusion matrix for FP, FN, TP and TN
# cm stands for confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
# plot
cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="RdPu", values_format="d")
disp.im_.colorbar.set_label("Number of peptides", fontsize=13) # color bar label

ax.set_xlabel("Predicted Class", fontsize=14)
ax.set_ylabel("True Class", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.set_title("SVM Confusion Matrix Pipeline A (HC50 features)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_CM, dpi=150)
plt.close()
print(f"Saved confusion matrix --> {OUT_CM}")
