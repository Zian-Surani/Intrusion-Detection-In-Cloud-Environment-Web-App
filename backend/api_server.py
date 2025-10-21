# FastAPI server for Sentinel X IDS
# This provides REST API endpoints for the React frontend to communicate with

import os
import time
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import requests
from urllib.parse import urlparse, urljoin
import urllib.robotparser

# Optional xgboost support
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

# Initialize FastAPI app
app = FastAPI(title="Sentinel X IDS API", version="1.0.0", description="Intrusion Detection System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "http://localhost:3000",
        "http://localhost:8080",  # Frontend Vite server port
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Data models for API requests/responses
class LogAnalysisRequest(BaseModel):
    logs: str
    use_neural_network: bool = True
    analysis_type: str = "upload"  # "upload" or "url"

class URLAnalysisRequest(BaseModel):
    url: str
    crawl_enabled: bool = False
    max_pages: int = 8
    crawl_delay: float = 0.4
    use_neural_network: bool = True

class AnalysisResponse(BaseModel):
    timestamp: str
    verdict: str
    method_used: str
    justification: str
    ann_prob: Optional[float]
    dfa_detail: Dict[str, List[int]]
    features: Dict[str, Any]
    plain_explanation: str
    line_details: List[Dict[str, Any]]

# Import all the analysis functions from the original Streamlit app
# DFA rules and analysis functions (copied from sentinelx_demo_app.py)

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

def extract_features(lines: List[str]) -> tuple[Dict[str, Any], List[float]]:
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

def build_full_feature_vector_from_small(feat_dict: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    """
    Construct a DataFrame with columns = expected_features (in order), filling missing columns with heuristics.
    """
    row = {col: 0.0 for col in expected_features}

    if "count" in expected_features and "count" in feat_dict:
        row["count"] = float(feat_dict.get("count", 0))

    if "avg_bytes" in feat_dict:
        avg_b = float(feat_dict.get("avg_bytes", 0.0))
        if "dst_bytes" in expected_features:
            row["dst_bytes"] = float(avg_b)
        if "total_bytes" in expected_features:
            row["total_bytes"] = float(avg_b) * max(1.0, float(feat_dict.get("count", 1)))

    if "src_bytes" in expected_features and "src_bytes" in feat_dict:
        row["src_bytes"] = float(feat_dict.get("src_bytes", 0.0))

    if "srv_count" in expected_features and "num_posts" in feat_dict:
        row["srv_count"] = int(feat_dict.get("num_posts", 0))

    if "dst_host_count" in expected_features:
        row["dst_host_count"] = int(min(255, max(1, feat_dict.get("count", 1) // 2)))
    if "dst_host_srv_count" in expected_features:
        row["dst_host_srv_count"] = int(min(255, max(1, feat_dict.get("count", 1) // 3)))

    if "same_srv_rate" in expected_features:
        row["same_srv_rate"] = float(feat_dict.get("same_srv_rate", 0.5))

    if "bytes_ratio" in expected_features:
        dst = row.get("dst_bytes", 0.0)
        src = row.get("src_bytes", 0.0)
        row["bytes_ratio"] = float(dst / (src + dst + 1e-6))

    prot_cols = [c for c in expected_features if c.startswith("protocol_type_")]
    if prot_cols:
        chosen = "protocol_type_SF" if "protocol_type_SF" in prot_cols else prot_cols[0]
        for c in prot_cols:
            row[c] = 1.0 if c == chosen else 0.0

    if "service" in expected_features:
        svc = feat_dict.get("service", "http")
        row["service"] = float(abs(hash(str(svc))) % 1000) / 1000.0
    if "flag" in expected_features:
        fl = feat_dict.get("flag", "SF")
        row["flag"] = float(abs(hash(str(fl))) % 1000) / 1000.0

    df = pd.DataFrame([{col: row.get(col, 0.0) for col in expected_features}])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def predict_with_booster_safe(booster, feat_dict: Dict[str, Any]) -> float:
    """Safely predict with a native xgboost.Booster"""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost not installed in environment")

    expected = None
    try:
        expected = list(booster.feature_names) if getattr(booster, "feature_names", None) is not None else None
    except Exception:
        expected = None

    if not expected or len(expected) == 0:
        try:
            expected = list(booster.attributes().get("feature_names", "").split(",")) if hasattr(booster, "attributes") else None
        except Exception:
            expected = None

    if not expected or len(expected) == 0:
        expected = list(feat_dict.keys())

    full_df = build_full_feature_vector_from_small(feat_dict, expected)
    full_df = full_df[expected]

    dmat = xgb.DMatrix(full_df.values.astype(np.float32), feature_names=expected)
    preds = booster.predict(dmat)
    preds = np.asarray(preds)
    if preds.ndim == 2:
        return float(preds[0, -1])
    return float(preds[0])

def infer_model(model_obj, feat_dict: Dict[str, Any]) -> Optional[float]:
    """Try to infer a single probability-like score (0..1) from various model object shapes."""
    X = pd.DataFrame([feat_dict])
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # Handle nested containers
    if isinstance(model_obj, dict):
        for key in ("model", "booster", "estimator", "best_estimator_", "clf", "pipeline"):
            if key in model_obj:
                cand = model_obj[key]
                return infer_model(cand, feat_dict)
    if isinstance(model_obj, (list, tuple)):
        for item in model_obj:
            if hasattr(item, "predict") or (HAS_XGBOOST and isinstance(item, xgb.Booster)):
                return infer_model(item, feat_dict)

    # sklearn-like predict_proba
    if hasattr(model_obj, "predict_proba"):
        try:
            p = model_obj.predict_proba(X)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return float(p[0, -1])
            if p.ndim == 1:
                return float(p[0])
        except Exception:
            pass

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
        except Exception:
            pass

    # xgboost models
    if HAS_XGBOOST:
        if hasattr(model_obj, "get_booster") and callable(getattr(model_obj, "get_booster")):
            try:
                if hasattr(model_obj, "predict_proba"):
                    p = model_obj.predict_proba(X)
                    p = np.asarray(p)
                    if p.ndim == 2 and p.shape[1] >= 2:
                        return float(p[0, -1])
                booster = model_obj.get_booster()
                return predict_with_booster_safe(booster, feat_dict)
            except Exception:
                pass

        if isinstance(model_obj, xgb.Booster):
            try:
                return predict_with_booster_safe(model_obj, feat_dict)
            except Exception:
                pass

    return None

def crawl_site(start_url: str, max_pages: int = 8, delay: float = 0.4):
    """Crawl a website and return log-like entries"""
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
            
            # Try to enqueue same-domain links using bs4 if available
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

def load_latest_model():
    """Load the latest model from artifacts directory"""
    try:
        files = [f for f in os.listdir(ARTIFACTS_DIR) if f.lower().endswith((".joblib", ".pkl"))]
        if not files:
            return None
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(ARTIFACTS_DIR, fn)), reverse=True)
        latest = os.path.join(ARTIFACTS_DIR, files[0])
        return joblib.load(latest)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Sentinel X IDS API", "version": "1.0.0", "xgboost_available": HAS_XGBOOST}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    """List available models in artifacts directory"""
    try:
        files = [f for f in os.listdir(ARTIFACTS_DIR) if f.lower().endswith((".joblib", ".pkl"))]
        models = []
        for f in files:
            path = os.path.join(ARTIFACTS_DIR, f)
            stat = os.stat(path)
            models.append({
                "filename": f,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/analyze/logs")
async def analyze_logs(request: LogAnalysisRequest) -> AnalysisResponse:
    """Analyze log data for security threats"""
    try:
        # Parse logs into lines
        lines = [l.strip() for l in request.logs.splitlines() if l.strip()]
        if not lines:
            raise HTTPException(status_code=400, detail="No valid log lines provided")

        # Load model if neural network analysis is requested
        model = None
        if request.use_neural_network:
            model = load_latest_model()

        # Run DFA analysis
        dfa_matched = dfa_check(lines, min_matches=1)

        # Extract features
        feat_dict, line_scores = extract_features(lines)

        # ANN inference
        ann_prob = None
        ann_used = False
        if model is not None and request.use_neural_network:
            try:
                ann_prob = infer_model(model, feat_dict)
                if ann_prob is not None:
                    ann_used = True
            except Exception as e:
                print(f"Model inference error: {e}")

        # Determine verdict
        ann_threshold = 0.7
        high_conf = 0.9
        
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
                # Heuristic fallback
                mean_score = float(np.mean(line_scores)) if line_scores else 0.0
                method_used = "HEURISTIC"
                justification = f"Heuristic mean line score = {mean_score:.2f}"
                if mean_score > 0.5:
                    verdict = "SUSPICIOUS (heuristic)"
                else:
                    verdict = "NORMAL (heuristic)"

        # Build line details
        line_details = []
        for i, (line, score) in enumerate(zip(lines, line_scores)):
            flags = []
            for rule_name, matched_indices in dfa_matched.items():
                if i in matched_indices:
                    flags.append("DFA")
                    score = max(score, 0.9)
            
            if ann_used and ann_prob is not None and ann_prob >= ann_threshold:
                suspect_tokens = [r'4\d{2}', r'5\d{2}', r'POST', r'login', r'auth', r'passwd', r'admin', r'wp-login', r'cmd', r'select', r'union', r'<script>']
                if any(re.search(tok, line, flags=re.I) for tok in suspect_tokens):
                    flags.append("ANN")
                    score = max(score, min(0.85, ann_prob))
            
            line_details.append({
                "line_number": i + 1,
                "content": line,
                "score": min(1.0, score),
                "flags": flags
            })

        # Create plain explanation
        top_lines = sorted(line_details, key=lambda x: x["score"], reverse=True)[:5]
        plain_explanation = f"""Verdict: {verdict}
Method used: {method_used}
Why: {justification}
Top suspicious lines:
{chr(10).join([f"- {line['content']}" for line in top_lines])}
How it works: (1) DFA quick patterns, (2) ANN model if available, (3) merge results to highlight suspicious lines."""

        # Save analysis result
        ts = int(time.time())
        result = {
            "timestamp": datetime.now().isoformat(),
            "verdict": verdict,
            "method_used": method_used,
            "justification": justification,
            "ann_prob": ann_prob,
            "dfa_detail": dfa_matched,
            "line_details": line_details,
            "features": feat_dict,
            "plain_explanation": plain_explanation
        }

        # Save to artifacts
        out_path = os.path.join(ARTIFACTS_DIR, f"result_{ts}.json")
        try:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Failed to save result: {e}")

        return AnalysisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze/url")
async def analyze_url(request: URLAnalysisRequest) -> AnalysisResponse:
    """Analyze a URL by crawling or fetching"""
    try:
        if request.crawl_enabled:
            lines, meta = crawl_site(request.url, max_pages=request.max_pages, delay=request.crawl_delay)
        else:
            resp = requests.get(request.url, timeout=10)
            status = resp.status_code
            content_length = resp.headers.get("Content-Length") or len(resp.content or b"")
            lines = [f'GET {request.url} HTTP/1.1" {status} Content-Length: {content_length}']

        if not lines:
            raise HTTPException(status_code=400, detail="No data retrieved from URL")

        # Convert to LogAnalysisRequest and analyze
        log_request = LogAnalysisRequest(
            logs="\n".join(lines),
            use_neural_network=request.use_neural_network,
            analysis_type="url"
        )
        
        return await analyze_logs(log_request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis error: {str(e)}")

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new model file"""
    if not file.filename.lower().endswith(('.joblib', '.pkl')):
        raise HTTPException(status_code=400, detail="Only .joblib and .pkl files are supported")
    
    try:
        ts = int(time.time())
        filename = f"uploaded_model_{ts}.joblib"
        file_path = os.path.join(ARTIFACTS_DIR, filename)
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Test loading the model
        try:
            joblib.load(file_path)
            return {"message": f"Model uploaded successfully as {filename}"}
        except Exception as e:
            os.remove(file_path)  # Clean up if model is invalid
            raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/recent-analyses")
async def get_recent_analyses():
    """Get recent analysis results"""
    try:
        files = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("result_") and f.endswith(".json")]
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(ARTIFACTS_DIR, fn)), reverse=True)
        
        recent = []
        for f in files[:10]:  # Get latest 10
            try:
                with open(os.path.join(ARTIFACTS_DIR, f), "r") as file:
                    data = json.load(file)
                    recent.append({
                        "id": f.replace("result_", "").replace(".json", ""),
                        "timestamp": data.get("timestamp", ""),
                        "verdict": data.get("verdict", ""),
                        "method_used": data.get("method_used", ""),
                        "line_count": len(data.get("line_details", []))
                    })
            except Exception:
                continue
        
        return {"analyses": recent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recent analyses: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)