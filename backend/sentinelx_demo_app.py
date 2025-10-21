# app.py
# Sentinel — Minimal robust Streamlit IDS with safe xgboost.Booster inference
#
# Features:
# - Detects sklearn pipelines/estimators, xgboost.XGBClassifier, and native xgboost.Booster
# - If Booster was trained on many columns (e.g. NSL-KDD features), builds a full feature vector
#   from the small runtime feature set by filling missing columns sensibly so Booster.predict works.
# - If no usable model, falls back to lightweight heuristics so the app never crashes.
# - Saves analysis JSON into ./artifacts/
#
# Usage:
# 1. pip install -r requirements.txt
#    requirements.txt should include: streamlit scikit-learn pandas numpy joblib matplotlib requests beautifulsoup4
#    If using xgboost Booster models: pip install xgboost
# 2. Put your model (joblib dump) into ./artifacts/ or upload with the UI.
# 3. streamlit run app.py
#
# Notes:
# - This approach of filling missing features is a demo fallback (not ideal for production).
# - For reproducible results, save a full sklearn Pipeline (preprocessing + estimator) at training time.

import os
import time
import json
import re
from typing import List, Dict, Any, Optional

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser

# Optional xgboost support (install if you plan to use Booster)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

st.set_page_config(page_title="Sentinel", layout="wide")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------
# Small DFA rules + feature extractor
# -----------------------
DFA_RULES = [
    ("4xx", re.compile(r'\" \b4\d{2}\b')),
    ("5xx", re.compile(r'\" \b5\d{2}\b')),
    ("POST", re.compile(r'\bPOST\b', flags=re.I)),
    ("small_resp", re.compile(r'Content-Length:\s*(?:[0-9]{1,2})\b', flags=re.I))
]

def dfa_check(lines: List[str], min_matches:int=1) -> Dict[str, List[int]]:
    matched = {}
    for name, pat in DFA_RULES:
        idxs = [i for i,l in enumerate(lines) if pat.search(l)]
        if len(idxs) >= min_matches:
            matched[name] = idxs
    return matched

def extract_features(lines: List[str]) -> (Dict[str, Any], List[float]):
    """Small numeric feature set used at runtime. Also returns per-line base severity scores."""
    count = len(lines)
    total_bytes = 0
    status_re = re.compile(r'\" (\d{3})\b')
    bytes_re = re.compile(r'Content-Length:\s*(\d+)', flags=re.I)

    line_scores = []
    for ln in lines:
        score = 0.0
        m = status_re.search(ln)
        if m:
            try:
                code = int(m.group(1))
                if 500 <= code < 600:
                    score += 0.7
                elif 400 <= code < 500:
                    score += 0.5
            except Exception:
                pass
        m2 = bytes_re.search(ln)
        b = 0
        if m2:
            try:
                b = int(m2.group(1))
            except Exception:
                b = 0
        total_bytes += b
        if b and b < 100:
            score += 0.15
        if re.search(r'POST', ln, flags=re.I):
            score += 0.25
        if re.search(r"(select\s+from|union\s+select|<script>)", ln, flags=re.I):
            score += 0.6
        line_scores.append(min(1.0, score))

    avg_bytes = float(total_bytes) / max(1, count)
    feats = {
        "count": int(count),
        "avg_bytes": float(avg_bytes),
        "num_posts": int(sum(1 for l in lines if re.search(r'\bPOST\b', l, flags=re.I)))
    }
    return feats, line_scores

# -----------------------
# Booster-safe helpers
# -----------------------
def _hash_to_float(x: Any) -> float:
    """Deterministic hash -> float in [0,1). Used as a fallback numeric encoding."""
    s = str(x)
    return (abs(hash(s)) % 10**8) / 10**8

def build_full_feature_vector_from_small(feat_dict: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    """
    Construct a DataFrame with columns = expected_features (in order), filling missing columns with heuristics.
    This allows a saved Booster (trained on many NSL-KDD-like features) to run on our small runtime features.
    """
    # initialize row with zeros
    row = {col: 0.0 for col in expected_features}

    # map small features into likely expected columns (heuristic)
    if "count" in expected_features and "count" in feat_dict:
        row["count"] = float(feat_dict.get("count", 0))

    # avg_bytes -> dst_bytes and total_bytes
    if "avg_bytes" in feat_dict:
        avg_b = float(feat_dict.get("avg_bytes", 0.0))
        if "dst_bytes" in expected_features:
            row["dst_bytes"] = float(avg_b)
        if "total_bytes" in expected_features:
            row["total_bytes"] = float(avg_b) * max(1.0, float(feat_dict.get("count", 1)))

    # src_bytes fallback from feat_dict if present
    if "src_bytes" in expected_features and "src_bytes" in feat_dict:
        row["src_bytes"] = float(feat_dict.get("src_bytes", 0.0))

    # srv_count from num_posts
    if "srv_count" in expected_features and "num_posts" in feat_dict:
        row["srv_count"] = int(feat_dict.get("num_posts", 0))

    # dst_host counts heuristics
    if "dst_host_count" in expected_features:
        row["dst_host_count"] = int(min(255, max(1, feat_dict.get("count", 1) // 2)))
    if "dst_host_srv_count" in expected_features:
        row["dst_host_srv_count"] = int(min(255, max(1, feat_dict.get("count", 1) // 3)))

    # same_srv_rate heuristic
    if "same_srv_rate" in expected_features:
        row["same_srv_rate"] = float(feat_dict.get("same_srv_rate", 0.5))

    # bytes_ratio heuristic
    if "bytes_ratio" in expected_features:
        dst = row.get("dst_bytes", 0.0)
        src = row.get("src_bytes", 0.0)
        row["bytes_ratio"] = float(dst / (src + dst + 1e-6))

    # protocol_type one-hot columns: set protocol_type_SF or any available to 1.0
    prot_cols = [c for c in expected_features if c.startswith("protocol_type_")]
    if prot_cols:
        chosen = "protocol_type_SF" if "protocol_type_SF" in prot_cols else prot_cols[0]
        for c in prot_cols:
            row[c] = 1.0 if c == chosen else 0.0

    # service and flag numeric fallbacks (coarse hash)
    if "service" in expected_features:
        svc = feat_dict.get("service", "http")
        row["service"] = float(abs(hash(str(svc))) % 1000) / 1000.0
    if "flag" in expected_features:
        fl = feat_dict.get("flag", "SF")
        row["flag"] = float(abs(hash(str(fl))) % 1000) / 1000.0

    # duration_* columns stay zero by default (already set)

    # convert to DataFrame in expected order
    df = pd.DataFrame([{col: row.get(col, 0.0) for col in expected_features}])
    # ensure numeric dtypes
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def predict_with_booster_safe(booster: "xgb.Booster", feat_dict: Dict[str, Any]) -> float:
    """
    Safely predict with a native xgboost.Booster by:
      - reading booster.feature_names
      - building a full input DataFrame matching those names (using heuristics)
      - converting to xgb.DMatrix and calling booster.predict
    Returns float prediction (single value). Raises exceptions on failure.
    """
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost not installed in environment")

    # Attempt to retrieve expected feature names in training order
    expected = None
    try:
        # booster.feature_names might be a list-like or None
        expected = list(booster.feature_names) if getattr(booster, "feature_names", None) is not None else None
    except Exception:
        expected = None

    if not expected or len(expected) == 0:
        # try attributes fallback
        try:
            # some boosters store attribute 'feature_names' as attribute map
            expected = list(booster.attributes().get("feature_names", "").split(",")) if hasattr(booster, "attributes") else None
        except Exception:
            expected = None

    if not expected or len(expected) == 0:
        # final fallback: use the provided feature keys only (prediction may fail if booster expected other names)
        expected = list(feat_dict.keys())

    # Build DataFrame with the expected features
    full_df = build_full_feature_vector_from_small(feat_dict, expected)

    # If number of columns/order doesn't match expected, ensure order matches exactly
    full_df = full_df[expected]

    # Convert to DMatrix and predict
    dmat = xgb.DMatrix(full_df.values.astype(np.float32), feature_names=expected)
    preds = booster.predict(dmat)
    preds = np.asarray(preds)
    if preds.ndim == 2:
        return float(preds[0, -1])
    return float(preds[0])

# -----------------------
# Robust infer_model wrapper
# -----------------------
def infer_model(model_obj, feat_dict: Dict[str, Any]) -> Optional[float]:
    """
    Try to infer a single probability-like score (0..1) from various model object shapes.
    Handles:
      - sklearn estimators / pipelines with predict_proba / predict
      - xgboost.XGBClassifier (sklearn wrapper)
      - xgboost.Booster (native) via predict_with_booster_safe
      - model wrapped inside dict/list with common keys
    Returns float or None.
    """
    X = pd.DataFrame([feat_dict])
    # convert potential object columns to numeric (small check)
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # helper to try nested containers
    if isinstance(model_obj, dict):
        for key in ("model", "booster", "estimator", "best_estimator_", "clf", "pipeline"):
            if key in model_obj:
                cand = model_obj[key]
                st.info(f"Found nested model under key '{key}' -> type {type(cand).__name__}")
                return infer_model(cand, feat_dict)
    if isinstance(model_obj, (list, tuple)):
        for item in model_obj:
            if hasattr(item, "predict") or (HAS_XGBOOST and isinstance(item, xgb.Booster)):
                st.info(f"Found candidate model in list/tuple element type {type(item).__name__}")
                return infer_model(item, feat_dict)

    # Log model summary for debugging
    try:
        st.write(f"Model type detected: {type(model_obj).__name__}")
        sample_attrs = [a for a in dir(model_obj) if not a.startswith("_")][:12]
        st.write("Model sample attributes:", sample_attrs)
    except Exception:
        pass

    # sklearn-like predict_proba
    if hasattr(model_obj, "predict_proba"):
        try:
            p = model_obj.predict_proba(X)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return float(p[0, -1])
            if p.ndim == 1:
                return float(p[0])
        except Exception as e:
            st.warning(f"predict_proba failed: {e}")

    # sklearn-like predict
    if hasattr(model_obj, "predict"):
        try:
            pred = model_obj.predict(X)
            pred = np.asarray(pred).reshape(-1)
            if pred.size >= 1:
                v = pred[0]
                try:
                    vf = float(v)
                    if vf in (0.0, 1.0):
                        return float(vf)
                    return float(1.0 / (1.0 + np.exp(-vf)))  # sigmoid mapping
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"predict failed: {e}")

    # xgboost sklearn wrapper or Booster
    if HAS_XGBOOST:
        # XGBClassifier wrapper (has get_booster)
        if hasattr(model_obj, "get_booster") and callable(getattr(model_obj, "get_booster")):
            try:
                st.info("Detected XGB sklearn-wrapper. Attempting predict_proba or underlying booster.")
                if hasattr(model_obj, "predict_proba"):
                    p = model_obj.predict_proba(X)
                    p = np.asarray(p)
                    if p.ndim == 2 and p.shape[1] >= 2:
                        return float(p[0, -1])
                booster = model_obj.get_booster()
                return predict_with_booster_safe(booster, feat_dict)
            except Exception as e:
                st.warning(f"XGB wrapper handling failed: {e}")

        # Native Booster
        if isinstance(model_obj, xgb.Booster):
            try:
                st.info("Detected native xgboost.Booster. Using safe booster prediction.")
                return predict_with_booster_safe(model_obj, feat_dict)
            except Exception as e:
                st.warning(f"xgboost.Booster prediction failed: {e}")

    # Nothing usable
    st.warning("Model object does not expose predict_proba/predict and is not an xgboost model.")
    return None

# -----------------------
# Crawler helper (small)
# -----------------------
def crawl_site(start_url: str, max_pages: int = 8, delay: float = 0.4):
    parsed = urlparse(start_url)
    base_netloc = parsed.netloc
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http(s) URLs allowed")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
    except Exception:
        rp = None
    ua = "SentinelXDemo/1.0"
    queue = [start_url]
    visited = set()
    lines = []
    pages = 0
    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        if rp and not rp.can_fetch(ua, url):
            continue
        try:
            resp = requests.get(url, timeout=6, headers={"User-Agent": ua})
            status = resp.status_code
            content_length = resp.headers.get("Content-Length")
            if content_length is None:
                content_length = len(resp.content) if resp.content is not None else 0
            lines.append(f'GET {resp.url} HTTP/1.1" {status} Content-Length: {content_length}')
            pages += 1
            # try to enqueue same-domain links using bs4 if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text or "", "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    joined = urljoin(resp.url, href)
                    p = urlparse(joined)
                    if p.netloc == base_netloc and p.scheme.lower() in ("http", "https"):
                        if joined not in visited and joined not in queue:
                            queue.append(joined)
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(delay)
    meta = {"requested": start_url, "pages_crawled": pages, "visited": len(visited)}
    return lines, meta

# -----------------------
# UI
# -----------------------
st.title("Sentinel X — IDS (Booster-safe inference)")
st.markdown("Paste logs or crawl an HTTP(S) URL. The app auto-loads the newest .joblib in ./artifacts/ if present.")

col_left, col_right = st.columns([1, 2])
with col_left:
    st.header("Model & settings")
    # find latest joblib
    latest = None
    try:
        files = [f for f in os.listdir(ARTIFACTS_DIR) if f.lower().endswith((".joblib", ".pkl"))]
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(ARTIFACTS_DIR, fn)), reverse=True)
        latest = os.path.join(ARTIFACTS_DIR, files[0]) if files else None
    except Exception:
        latest = None

    if latest:
        st.write("Auto-detected model:", os.path.basename(latest))
    else:
        st.write("No model found in artifacts/. You can upload one below or drop a .joblib into artifacts/")

    uploaded = st.file_uploader("Upload a joblib/pkl model (optional)", type=["joblib","pkl"])
    model = None
    model_path = None
    if uploaded is not None:
        ts = int(time.time())
        savep = os.path.join(ARTIFACTS_DIR, f"uploaded_model_{ts}.joblib")
        with open(savep, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Uploaded and saved to {savep}")
        model_path = savep
        model = joblib.load(savep)
    else:
        if latest:
            try:
                model = joblib.load(latest)
                model_path = latest
            except Exception as e:
                st.warning(f"Could not load auto-detected model: {e}")
                model = None

    st.markdown("---")
    ann_threshold = st.slider("ANN suspicious threshold", 0.0, 1.0, 0.7, 0.05)
    high_conf = st.slider("ANN high confidence threshold", 0.0, 1.0, 0.9, 0.01)

    st.markdown("---")
    st.write("xgboost installed:", HAS_XGBOOST)
    if model_path:
        st.write("Loaded model path:", model_path)

with col_right:
    st.header("Input")
    mode = st.radio("Input mode", ("Paste logs", "Enter HTTP URL (demo)"))
    lines: List[str] = []
    if mode == "Paste logs":
        raw = st.text_area("Paste HTTP-style logs (one per line). Example:\nGET /index.html HTTP/1.1\" 200 Content-Length: 1200\nPOST /login HTTP/1.1\" 403 Content-Length: 45", height=260)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    else:
        url = st.text_input("Enter an HTTP(S) URL (http:// or https://)")
        crawl_enabled = st.checkbox("Crawl site to generate multi-line logs (same-domain)")
        max_pages = st.slider("Max pages to crawl", 1, 50, 8)
        crawl_delay = st.number_input("Delay between requests (seconds)", value=0.4, min_value=0.0, step=0.1, format="%.2f")
        if st.button("Fetch URL") and url:
            try:
                if crawl_enabled:
                    with st.spinner("Crawling site..."):
                        lines, meta = crawl_site(url, max_pages=max_pages, delay=float(crawl_delay))
                    if lines:
                        st.success(f"Crawled {meta['pages_crawled']} pages; visited {meta['visited']} URLs.")
                    else:
                        st.info("Crawl returned no pages (robots.txt or blocked).")
                else:
                    resp = requests.get(url, timeout=10)
                    status = resp.status_code
                    content_length = resp.headers.get("Content-Length") or len(resp.content or b"")
                    line = f'GET {url} HTTP/1.1" {status} Content-Length: {content_length}'
                    lines = [line]
                    st.success(f"Fetched {url} - status {status}")
                    st.code(line)
            except Exception as e:
                st.error(f"Fetch/crawl failed: {e}")

    if not lines:
        st.info("No input lines yet.")
    else:
        st.subheader("Input preview (first 200 lines)")
        for i, ln in enumerate(lines[:200]):
            st.text(f"{i+1}: {ln}")

# -----------------------
# Run analysis
# -----------------------
if st.button("Run IDS Analysis"):
    if not lines:
        st.error("No input provided.")
    else:
        # DFA rules
        dfa_matched = dfa_check(lines, min_matches=1)

        # extract features
        feat_dict, line_scores = extract_features(lines)
        st.write("Runtime features prepared:", feat_dict)

        # ANN inference (robust)
        ann_prob = None
        ann_used = False
        if model is not None:
            try:
                ann_prob = infer_model(model, feat_dict)
                if ann_prob is not None:
                    ann_used = True
                    st.write("ANN score:", ann_prob)
                else:
                    st.info("Model loaded but no usable prediction returned; falling back to heuristic.")
            except Exception as e:
                st.warning(f"Model inference error: {e}")
                ann_prob = None
                ann_used = False

        # Decide verdict
        if dfa_matched:
            verdict = "ATTACK (DFA)"
            method_used = "DFA"
            justification = f"DFA matched rules: {list(dfa_matched.keys())}"
        else:
            if ann_used and ann_prob is not None:
                method_used = "ANN"
                justification = f"ANN probability = {ann_prob:.3f}"
                if ann_prob >= high_conf:
                    verdict = f"ATTACK (ANN prob={ann_prob:.2f})"
                elif ann_prob >= ann_threshold:
                    verdict = f"SUSPICIOUS (ANN prob={ann_prob:.2f})"
                else:
                    verdict = f"NORMAL (ANN prob={ann_prob:.2f})"
            else:
                # heuristic fallback using mean line_scores
                mean_score = float(np.mean(line_scores)) if line_scores else 0.0
                method_used = "HEURISTIC"
                justification = f"Heuristic mean line score = {mean_score:.2f}"
                if mean_score > 0.5:
                    verdict = "SUSPICIOUS (heuristic)"
                else:
                    verdict = "NORMAL (heuristic)"

        # Build DataFrame of lines with scores/flags
        df_lines = pd.DataFrame({"line": lines, "base_score": line_scores})
        df_lines["dfa_flag"] = False
        for idxs in dfa_matched.values():
            for idx in idxs:
                if 0 <= idx < len(df_lines):
                    df_lines.at[idx, "dfa_flag"] = True
                    df_lines.at[idx, "base_score"] = max(df_lines.at[idx, "base_score"], 0.9)
        df_lines["ann_flag"] = False
        if ann_used and ann_prob is not None and ann_prob >= ann_threshold:
            suspect_tokens = [r'4\d{2}', r'5\d{2}', r'POST', r'login', r'auth', r'passwd', r'admin', r'wp-login', r'cmd', r'select', r'union', r'<script>']
            for i, ln in enumerate(df_lines["line"]):
                if any(re.search(tok, ln, flags=re.I) for tok in suspect_tokens):
                    df_lines.at[i, "ann_flag"] = True
                    df_lines.at[i, "base_score"] = max(df_lines.at[i, "base_score"], min(0.85, ann_prob))
        df_lines["score"] = df_lines["base_score"].clip(0.0, 1.0)

        # Plain explanation
        top_lines = list(df_lines.sort_values("score", ascending=False).head(6)["line"].values)
        plain_expl = "\n".join([
            f"Verdict: {verdict}",
            f"Method used: {method_used}",
            f"Why: {justification}",
            "Top suspicious lines:",
            *[f"- {l}" for l in top_lines[:5]],
            "How it works: (1) DFA quick patterns, (2) ANN model if available, (3) merge results to highlight suspicious lines."
        ])

        # Display results
        st.markdown("## Final verdict")
        if "ATTACK" in verdict:
            st.error(verdict)
        elif "SUSPICIOUS" in verdict:
            st.warning(verdict)
        else:
            st.success(verdict)

        st.markdown("### Justification (technical)")
        st.write(justification)

        st.markdown("### Plain explanation (non-technical)")
        st.text(plain_expl)

        st.markdown("### Line-level highlights")
        for i, row in df_lines.iterrows():
            score = row["score"]
            tags = []
            if row["dfa_flag"]:
                tags.append("DFA")
            if row["ann_flag"]:
                tags.append("ANN")
            tag_text = " & ".join(tags) if tags else "None"
            if score >= 0.8:
                bgcolor = "#ffcccc"
            elif score >= 0.5:
                bgcolor = "#fff2cc"
            elif score >= 0.2:
                bgcolor = "#e6f7ff"
            else:
                bgcolor = "white"
            st.markdown(
                f"<div style='background:{bgcolor};padding:6px;border-radius:6px;margin-bottom:4px'>"
                f"<b>{i+1}</b>: {row['line']} <span style='float:right'>Score: {score:.2f} | Flags: {tag_text}</span></div>",
                unsafe_allow_html=True
            )

        # Heatmap with black labels
        try:
            fig, ax = plt.subplots(figsize=(min(14, max(6, len(df_lines) * 0.12)), 2))
            im = ax.imshow(np.array(df_lines["score"].values).reshape(1, -1), aspect='auto', cmap='Reds', vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xticks(range(len(df_lines)))
            ax.set_xticklabels([str(i+1) for i in range(len(df_lines))], rotation=90, fontsize=8, color='black')
            ax.set_title("Per-line severity heatmap", color='black')
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label('Severity score (0-1)', color='black')
            cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
            for spine in ax.spines.values():
                spine.set_color('black')
            ax.tick_params(axis='x', colors='black')
            st.pyplot(fig)
        except Exception as e:
            st.warning("Heatmap generation failed: " + str(e))

        # Save analysis to artifacts
        ts = int(time.time())
        result = {
            "timestamp": ts,
            "verdict": verdict,
            "method_used": method_used,
            "justification": justification,
            "ann_prob": ann_prob,
            "dfa_detail": dfa_matched,
            "lines": df_lines.to_dict(orient="records"),
            "features": feat_dict,
            "plain_explanation": plain_expl
        }
        out_path = os.path.join(ARTIFACTS_DIR, f"result_{ts}.json")
        try:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            st.success(f"Saved analysis result to {out_path}")
        except Exception as e:
            st.warning("Failed to save result: " + str(e))

        # Provide download button for JSON
        st.download_button("Download analysis JSON", json.dumps(result, indent=2), file_name=f"sentinelx_result_{ts}.json")

st.markdown("---")
st.markdown("**Reminder:** only crawl/test websites you own or have permission to test. Do not paste real credentials or private keys.")
