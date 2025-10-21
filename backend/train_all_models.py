# train_all_models.py
import os, io, time, json, requests
import numpy as np, pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def download_nsl_kdd():
    try:
        cols_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Field Names.csv"
        tr_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
        te_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
        c = requests.get(cols_url, timeout=30)
        col_lines = [ln.strip() for ln in c.text.splitlines() if ln.strip()]
        feature_cols = [ln.split(",")[0] for ln in col_lines]
        rtr = requests.get(tr_url, timeout=60)
        rte = requests.get(te_url, timeout=60)
        df_tr = pd.read_csv(io.StringIO(rtr.text), header=None, names=feature_cols)
        df_te = pd.read_csv(io.StringIO(rte.text), header=None, names=feature_cols)
        df = pd.concat([df_tr, df_te], ignore_index=True)
        return df
    except Exception as e:
        print("NSL-KDD download failed:", e)
        return None

def prepare_dataframe(df):
    label_candidates = ["label","class","attack","dst_host_rerror_rate","type"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]
    df = df.rename(columns={label_col: "label"})
    keep = [c for c in ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
                        "count","srv_count","same_srv_rate","dst_host_count","dst_host_srv_count","label"] if c in df.columns]
    df2 = df[keep].dropna()
    df2["binary"] = (df2["label"].astype(str).str.lower() != "normal").astype(int)
    return df2

def train_models(force_synthetic=False, n_samples_limit=None):
    df = None
    if not force_synthetic:
        df = download_nsl_kdd()
    if df is None:
        np.random.seed(7)
        n = n_samples_limit or 8000
        df = pd.DataFrame({
            "duration": np.random.exponential(2.0, n),
            "protocol_type": np.random.choice(["tcp","udp","icmp"], n),
            "service": np.random.choice(["http","ftp","smtp","dns","ssh","other"], n),
            "flag": np.random.choice(["SF","S0","REJ","RSTR","SH"], n),
            "src_bytes": np.random.randint(0,10000,n),
            "dst_bytes": np.random.randint(0,10000,n),
            "count": np.random.randint(1,100,n),
            "srv_count": np.random.randint(1,100,n),
            "same_srv_rate": np.random.rand(n),
            "dst_host_count": np.random.randint(1,255,n),
            "dst_host_srv_count": np.random.randint(1,255,n),
            "label": np.random.choice(["normal","dos","probe"], n, p=[0.6,0.3,0.1])
        })
    df2 = prepare_dataframe(df)
    X = df2.drop(columns=["label","binary"])
    y = df2["binary"]
    if n_samples_limit:
        X = X.iloc[:n_samples_limit]; y = y.iloc[:n_samples_limit]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preproc = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)], remainder="drop")
    mlp = MLPClassifier(hidden_layer_sizes=(256,128), early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=200, random_state=42)
    pipe_mlp = Pipeline([("preproc", preproc), ("clf", mlp)])
    pipe_mlp.fit(X_train, y_train)
    joblib.dump(pipe_mlp, os.path.join(ARTIFACT_DIR, f"mlp_colab_{int(time.time())}.joblib"))
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    pipe_rf = Pipeline([("preproc", preproc), ("clf", rf)])
    pipe_rf.fit(X_train, y_train)
    joblib.dump(pipe_rf, os.path.join(ARTIFACT_DIR, f"rf_colab_{int(time.time())}.joblib"))
    return True

if __name__ == "__main__":
    train_models(force_synthetic=True, n_samples_limit=4000)
