#!/bin/sh
set -eu

PORT="${PORT:-10000}"
python /app/healthcheck.py
echo "Starting Streamlit on 0.0.0.0:${PORT}"
exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
