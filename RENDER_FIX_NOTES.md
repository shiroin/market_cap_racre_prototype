# Render timeout fix

The previous Dockerfile started `multi_chart_app.py` directly. This fix restores `app.py` as the service entrypoint and makes startup observable and deterministic.

Changes:
- explicit `start.sh`
- `python -m streamlit` instead of relying on the executable path
- bind to `0.0.0.0:$PORT`
- disable Streamlit file watching in production
- startup diagnostics for Python / Streamlit / FFmpeg / PORT
- Render health check at `/_stcore/health`
- smaller Docker build context

After merge, the Render log should print the diagnostics and `Starting Streamlit...` immediately before Streamlit starts.
