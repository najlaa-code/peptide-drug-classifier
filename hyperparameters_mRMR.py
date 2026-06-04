"""
Src: https://pmc.ncbi.nlm.nih.gov/articles/PMC12848716/
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("features_scaled.csv")  # pre-mRMR, all ~1529 features

META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic"]
feature_columns = [col for col in df.columns if col not in META_COLUMNS]

X = df[feature_columns].values
y = df["Class"].values  # 4-class labels

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),  # runs inside each fold
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

print(f"Best k:                {grid_search.best_params_['feature_selection__k']}")
print(f"Best C:                {grid_search.best_params_['classifier__C']}")
print(f"Best gamma:            {grid_search.best_params_['classifier__gamma']}")
print(f"Best balanced accuracy: {grid_search.best_score_:.4f}")
