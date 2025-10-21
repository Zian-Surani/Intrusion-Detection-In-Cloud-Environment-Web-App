# Sentinel X - Demo Package

This package contains a Streamlit demo app and a simple training script for the Sentinel X IDS project.

## Files
- `sentinelx_demo_app.py` - Streamlit demo. Paste HTTP-style logs or fetch an HTTP URL (demo). Upload your trained sklearn `.joblib` model into `artifacts/` and update `MODEL_PATH` or upload it using the UI.
- `train_all_models.py` - Training helper script (downloads NSL-KDD if available or uses synthetic data). Run in Colab or locally to produce model artifacts.
- `requirements.txt` - Python dependencies.
- `artifacts/` - folder to place your trained models (not included).
- `google_sample.json` - small sample HAR-like JSON for demo (included).

## How to run (locally)
1. Create a virtualenv and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place your trained model `.joblib` into `artifacts/` and set `MODEL_PATH` variable in `sentinelx_demo_app.py`.
3. Run:
   ```
   streamlit run sentinelx_demo_app.py
   ```

## Notes
- The demo app uses heuristic feature extraction for pasted logs. For production, replace `extract_features_from_lines` with your real feature extraction/sessionizer logic.
- Use HTTPS capture (Playwright) or HAR files for richer input.
