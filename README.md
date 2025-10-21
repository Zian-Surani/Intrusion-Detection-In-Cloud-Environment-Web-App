# Intrusion Detection in Cloud Environment Web App  
*A full‑stack web application for intrusion detection in cloud/log environments using regex/DFA & log‑analysis.*

---

## 📘 Overview  
This repository contains a comprehensive web application designed to detect intrusions in a cloud or web‑log environment. It uses pattern matching (regular expressions, deterministic finite automata) and log‐analysis to detect anomalous or malicious inputs. This is useful for cloud‑based services, logging infrastructures, web servers, and security‑analytics dashboards.

The key components:  
- **Backend**: Implements log ingestion, pattern matching (regex/DFA), anomaly detection logic.  
- **Web Frontend**: Allows uploading logs or entering HTTP request lines, and visualises alerts/detections.  
- **Cloud‑Ready Architecture**: Designed to work in a cloud environment (containerised or VM), with scalability in mind.  
- **Logging & Alerting**: Records results of detection, supports analysis of false positives/negatives.

---

## 🧩 Key Features  
- **Log & HTTP Request Monitoring**: Accepts web server logs, HTTP request entries, and scans them for intrusion patterns.  
- **Regex & DFA‑based Detection**: Uses a rule set built with regular expressions and deterministic finite automata (DFA) to identify malicious payloads or patterns.  
- **Cloud‑Environment Focus**: Architecture built to be deployed on cloud infrastructures, supports scale, logging, and multi‑tenant scenarios.  
- **Web UI for Live Monitoring**: Users can input log snippets or files and immediately view detection results and alerts.  
- **Modular & Extendable**: Detection rules, logging, UI can all be adapted or extended for new threat types.

---

## 📁 Repository Structure  
```
/
├── backend/                    # Core backend logic
│   ├── app.py                  # Main backend service (e.g., Flask/FastAPI)
│   ├── rules/                  # Regex/DFA rule definitions for intrusion patterns
│   ├── detectors/              # Modules implementing pattern matching logic
│   └── requirements.txt        # Python dependencies
├── frontend/                   # Web UI
│   ├── index.html              # Frontend HTML page
│   ├── static/                 # CSS/JS assets
│   └── main.js                 # UI logic
├── Result Analysis/            # Analysis of this system with existing analyzers
│   ├── ANALYSIS.m              # MATLAB code
│   ├── png files               # MATLAB output
├── tests/                      # Test suites (optional)
├── README.md                   # This file
└── .gitignore                  # Ignore settings
```

---

## ⚙️ Installation & Setup  
### Prerequisites  
- Python 3.8+  
- pip or conda  
- (Optional) A cloud VM/container environment (Docker, Kubernetes)  
- A modern browser for UI interface  

### Backend Setup  
```bash
cd backend
python -m venv .venv
# For Windows:
.\.venv\Scriptsctivate
# For macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the Backend Server  
```bash
cd backend
# Example (Flask):
flask run --host=0.0.0.0 --port=5000
```
Or if using FastAPI:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup  
Open a browser and open `frontend/index.html` (or serve via HTTP server).  
Ensure the backend service URL is correctly configured in `main.js` or UI code.

---

## 🧠 How It Works  
### 1. Log Ingestion & Pre‑processing  
- The backend accepts log lines (HTTP request logs, access logs) or raw HTTP request entries.  
- Pre‑processing: Normalises whitespace, extracts key fields (method, URI, headers), tokenises input where needed.

### 2. Pattern Matching (Regex/DFA)  
- A rule set (defined in `backend/rules/`) consists of regular expressions and DFA patterns representing common attack vectors (e.g., SQL injection, XSS, path traversal, malicious user‑agent strings).  
- The detectors implement a matching engine: they check each input against each rule, possibly in a DFA representation for performance in high‑throughput scenarios.

### 3. Alert Generation & Logging  
- If a match is found, an alert is generated with:  
  - Type of intrusion (based on rule)  
  - Severity level  
  - Matched input snippet  
- Alerts are logged into a persistent store (file, database, or cloud logging service) for later review.

### 4. Web UI Visualisation  
- Users can upload log files or enter live log lines via the UI.  
- The UI shows:  
  - Detection results (alerts, matched rules)  
  - Summary metrics (number of matches, types, severities)  
  - Possibly metrics like false positives/negatives if ground truth is available.

### 5. Cloud Deployment Considerations  
- Backend can run inside a container (Docker) and scale horizontally behind a load‑balancer.  
- Logging can integrate with cloud logging services (e.g., AWS CloudWatch, GCP Stackdriver) for centralised alerting.  
- Rules can be updated dynamically (via config reload) to adapt to evolving threats.

---

## 🧪 Example Usage  
**Scenario:** A web server log includes a malicious request:  
```
GET /index.php?id=1; DROP TABLE users; -- HTTP/1.1
User‑Agent: evilBot/1.0
```

**Steps:**  
1. Upload this log line via the UI or send it to the backend API.  
2. The detector engine matches the `DROP TABLE` pattern via regex.  
3. An alert is created:  
   - Type: SQL Injection  
   - Severity: High  
   - Matched Rule: `.*DROP\s+TABLE.*`  
4. UI shows the alert in real time, logs stored for review.

---

## 📋 API Reference (if applicable)  
If the backend exposes REST endpoints, they may look like:

| Endpoint             | Method | Description                                               |
|----------------------|--------|-----------------------------------------------------------|
| `/api/logs/analyze`  | POST   | Accepts log lines (or a file) → returns detection results |
| `/api/rules/list`    | GET    | Returns list of active detection rules                     |
| `/api/alerts`        | GET    | Returns recent alerts and summaries                        |

**Example Request:**
```json
{
  "log_lines": [
    "GET /foo?param=1 HTTP/1.1",
    "GET /index.php?id=1; DROP TABLE users; -- HTTP/1.1"
  ]
}
```

**Example Response:**
```json
{
  "alerts": [
    {
      "line": "GET /index.php?id=1; DROP TABLE users; -- HTTP/1.1",
      "rule": "SQL_INJECTION_01",
      "severity": "High",
      "matched_text": "DROP TABLE users"
    }
  ]
}
```

---

## 🎯 Use Cases  
- **Cloud Service Providers (CSPs):** Monitor VMs/containers for suspicious HTTP traffic or logs.  
- **Web Application Security Teams:** Provide a lightweight intrusion detection front‑end for web logs without heavy SIEM overhead.  
- **DevOps / SRE Teams:** Hook into CI/CD pipelines to scan logs from staging/production for early intrusion detection.  
- **Research / Education:** Use the application as a reference implementation of log‑based intrusion detection using regex/DFA.

---

## 🛠️ Customisation & Extension  
- **Add/Modify Detection Rules:** Edit or add regex/DFA rules under `backend/rules/` for new attack types.  
- **Extend UI:** Enhance the frontend to provide dashboards, threat visualisations, and analytics (e.g., alerts over time, top rule hits).  
- **Integrate ML/Anomaly Detection:** In addition to regex/DFA, integrate machine learning models to detect unknown or zero‑day attacks.  
- **Cloud Native Improvements:** Containerise backend, add Kubernetes manifests, enable auto‑scaling and cloud logging.  
- **Alerting/Response Integration:** Connect with email, Slack, SIEM, or trigger automated responses (e.g., block IPs) on high severity alerts.

---

## 📜 License  
```
MIT License  
Copyright (c) 2025
```
*(Please ensure to update year/author as appropriate.)*

---

## 🤝 Contribution  
Contributions are welcome!  
1. Fork the repository and create a feature branch (`feature‑xyz`).  
2. Implement your changes and add/update tests.  
3. Submit a Pull Request with a clear description of the change.

Please follow code style, update this documentation if your changes affect it, and include relevant tests.

---

## 👨‍💻 Contributors  
- **Vaishnavi Soni** – Lead Developer  
- **Rituraj Kushwaha** – Research Analyst  
- **Zian Rajeshkumar Surani** – Web Developer & System Tester  

---

**Developed with ❤️ by the Intrusion Detection Team**  
> “Enabling robust intrusion detection in the cloud through pattern‑based analysis.”
