FROM python:3.11-slim

# FFmpeg + 日本語フォント
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

# Render assigns PORT at runtime. Use the lightweight root app as the
# Streamlit entrypoint so pages/ provides native multipage navigation.
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
