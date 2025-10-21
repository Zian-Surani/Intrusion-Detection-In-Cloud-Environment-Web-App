# train_small_models.py
# Train two small models on synthetic data that matches the runtime features
# Saves: artifacts/ann_small_xgb.joblib and artifacts/rf_small.joblib
# Requirements: scikit-learn, xgboost, pandas, numpy, joblib

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

# --------------- Create synthetic dataset that uses the same small features your app prepares -------------
# Features: count (int), avg_bytes (float), num_posts (int)
# Label: attack (1) if many posts or very small average bytes or many lines with 5xx/4xx (simulated)
N = 20000
rng = np.random.RandomState(42)

count = rng.poisson(lam=6, size=N)               # typical number of lines
avg_bytes = rng.exponential(scale=400.0, size=N) # average response size
num_posts = rng.binomial(n=10, p=0.12, size=N)   # how many POST requests in session

# Create synthetic label with some rules + noise
label = ((num_posts >= 2) | (avg_bytes < 120) | (count > 20)).astype(int)
# add some noise so not perfect
noise_idx = rng.choice(N, size=int(0.03 * N), replace=False)
label[noise_idx] = 1 - label[noise_idx]

X = pd.DataFrame({
    "count": count,
    "avg_bytes": avg_bytes,
    "num_posts": num_posts
})
y = label

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# ---------------- RandomForest pipeline (sklearn) ----------------
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])
print("Training RandomForest pipeline...")
rf_pipe.fit(X_train, y_train)
proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, proba_rf)
print(f"RandomForest AUC: {auc_rf:.4f}")
joblib.dump(rf_pipe, os.path.join(ART, "rf_small.joblib"))

# ---------------- XGBoost classifier (sklearn wrapper) ----------------
# XGBClassifier gives predict_proba and is safe for the Streamlit wrapper
xgb_clf = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=4)
xgb_pipe = Pipeline([
    ("scaler", StandardScaler()),  # keep a scaler for numeric stability
    ("xgb", xgb_clf)
])
print("Training XGBClassifier pipeline...")
xgb_pipe.fit(X_train, y_train)
proba_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, proba_xgb)
print(f"XGBClassifier AUC: {auc_xgb:.4f}")
joblib.dump(xgb_pipe, os.path.join(ART, "ann_small_xgb.joblib"))

# Print classification reports for sanity
yhat_rf = (proba_rf >= 0.5).astype(int)
yhat_xgb = (proba_xgb >= 0.5).astype(int)
print("\nRF classification report (test):")
print(classification_report(y_test, yhat_rf))
print("\nXGB classification report (test):")
print(classification_report(y_test, yhat_xgb))

print("\nSaved models to artifacts/: rf_small.joblib and ann_small_xgb.joblib")
