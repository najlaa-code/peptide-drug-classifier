"""
Pipeline A - Hyperparameter Search
Src: https://pmc.ncbi.nlm.nih.gov/articles/PMC12848716/
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
df = pd.read_csv("features_scaled_A.csv")  # output of pipeline_A_scaling.py
META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic", "Split"]
CLASS_NAMES = ["Inactive", "Selective", "Pure hemolytic", "Both active/toxic"]
# restrict to the train split only for pipeline A
if "Split" not in df.columns:
    raise ValueError("Expected a 'Split' column from pipeline_A_step0_split.py — got none.")
train_df = df[df["Split"] == "train"].reset_index(drop=True)
print(f"Training rows: {len(train_df)} (of {len(df)} total; "
      f"{(df['Split'] == 'test').sum()} test rows held out and untouched)")

feature_columns = [col for col in train_df.columns if col not in META_COLUMNS]
X = train_df[feature_columns].values
y = train_df["Class"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', SVC(kernel='rbf', class_weight='balanced'))
])
param_grid = {
    'feature_selection__k': [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000],
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2,
    refit=True
)
grid_search.fit(X, y)
print(f"Best k:{grid_search.best_params_['feature_selection__k']}")
print(f"Best C:{grid_search.best_params_['classifier__C']}")
print(f"Best gamma:{grid_search.best_params_['classifier__gamma']}")
print(f"Best CV balanced accuracy (train only, model selection — not generalization): {grid_search.best_score_:.4f}")
best = grid_search.best_estimator_
y_pred = cross_val_predict(best, X, y, cv=cv)
report = classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0)
print("\nPer-class report (cross_val_predict, TRAIN partition only):")
print(report)
output_lines = [
    "GridSearchCV - SVM Classification (4-class AMP) - Pipeline A",
    "Src: https://pmc.ncbi.nlm.nih.gov/articles/PMC12848716/",
    "NOTE: all results below are computed on the TRAIN partition only (Split == 'train').",
    "They represent model-selection performance, not a generalization estimate.",
    "The held-out test partition is scored separately in the final evaluation step.",
    "",
    f"Best k: {grid_search.best_params_['feature_selection__k']}",
    f"Best C: {grid_search.best_params_['classifier__C']}",
    f"Best gamma: {grid_search.best_params_['classifier__gamma']}",
    f"Best CV balanced accuracy (train only): {grid_search.best_score_:.4f}",
    "",
    "Per-class report (cross_val_predict, train only):",
    report,
]
with open("classification_gridsearch_results_pipelineA.txt", "w") as f:
    f.write("\n".join(output_lines))
