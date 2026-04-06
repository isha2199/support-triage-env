# ── Customer Support Ticket Triage — OpenEnv ─────────────────────────────────
# Builds a FastAPI server exposed on port 7860 (Hugging Face Spaces standard).
#
# Build:  docker build -t cstt-env .
# Run:    docker run -p 7860:7860 cstt-env
# Test:   curl http://localhost:7860/

FROM python:3.11-slim

# Security: run as non-root
RUN useradd --create-home appuser
WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment/ ./environment/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .

# Ensure python path includes /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
