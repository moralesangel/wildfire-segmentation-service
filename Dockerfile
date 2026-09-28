# Serving image. Hugging Face Spaces runs this directly.
FROM python:3.11-slim

# libgomp is ONNX Runtime's OpenMP runtime: without it the import fails at
# startup with an error that does not name the missing library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing code does not invalidate the pip layer.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY models/ models/
COPY static/ static/

# Spaces serves on 7860. Matching it here means the same image runs in both
# places without an override.
ENV PORT=7860 \
    WFS_MODEL_PATH=models/unet_fire.onnx \
    WFS_ONNX_THREADS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# A non-root user: the process never needs to write outside /tmp.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# One worker on purpose. The batcher owns the ONNX session and serialises
# inference; a second worker would mean two sessions competing for the same
# two vCPUs, which is slower than one and doubles the memory.
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
