import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv("features_scaled_2.csv")
META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic"]
feature_columns = [col for col in df.columns if col not in META_COLUMNS]
X = df[feature_columns].values
y_class = df["Class"].values
y_reg = df["log10hc50"].values
#y_reg = df["log10mic"].values

k_options = [100, 200, 400, 600]
C_options = [0.1, 1, 10]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = {}

for k in k_options:
    for C in C_options:
        fold_scores = []

        for train_idx, test_idx in cv.split(X, y_class):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_reg, _ = y_reg[train_idx], y_reg[test_idx]
            y_train_class, y_test_class = y_class[train_idx], y_class[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            selector = SelectKBest(score_func=f_regression, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train_reg)
            X_test_selected = selector.transform(X_test_scaled)

            clf = SVC(kernel='rbf', C=C, class_weight='balanced', random_state=42)
            clf.fit(X_train_selected, y_train_class)

            preds = clf.predict(X_test_selected)
            score = balanced_accuracy_score(y_test_class, preds)
            fold_scores.append(score)

        avg_cv_score = np.mean(fold_scores)
        print(f"Tested: k={k}, C={C} - Mean CV Balanced Accuracy: {avg_cv_score:.4f}")

        if avg_cv_score > best_score:
            best_score = avg_cv_score
            best_params = {'k': k, 'C': C}

print(f"Best parameters found: {best_params}")
print(f"Best CV balanced accuracy: {best_score:.4f}")
