# Serving image for the FastAPI search API (CPU). Mount a prebuilt index +
# manifest at /app/data; CLIP weights download on first run.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU torch first (smaller), then the rest of the serving stack
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    open_clip_torch faiss-cpu fastapi "uvicorn[standard]" python-multipart \
    pillow numpy pandas pyarrow pyyaml

COPY src ./src
COPY configs ./configs

EXPOSE 8000
ENV MMSEARCH_CONFIG=configs/serve.yaml
# docker run -p 8000:8000 -v "$PWD/data:/app/data" multimodal-image-search
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
