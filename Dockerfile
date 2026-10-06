FROM python:3.11-slim

# FFmpeg + 日本語フォント
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "streamlit run multi_chart_app.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true"]
