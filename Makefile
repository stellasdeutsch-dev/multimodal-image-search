CONFIG_DIR ?= configs

.PHONY: help setup sample ingest embed index serve ui eval test lint smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package with all extras
	python -m pip install -e ".[all]"

sample:  ## Generate a tiny synthetic image set (no download)
	python scripts/make_sample_images.py --out data/images --n 24

ingest:  ## Phase 1: build the manifest
	python -m src.ingest --config $(CONFIG_DIR)/data.yaml

embed:  ## Phase 2: batch CLIP embedding (GPU)
	python -m src.embed --config $(CONFIG_DIR)/embed.yaml

index:  ## Phase 3: build the vector index
	python -m src.index --config $(CONFIG_DIR)/index.yaml --build

serve:  ## Phase 4: run the FastAPI service
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

ui:  ## Phase 6: run the Gradio web UI
	python web/app.py

eval:  ## Phase 7: evaluate retrieval quality
	python -m eval.evaluate --config $(CONFIG_DIR)/serve.yaml --queries eval/queries.example.jsonl

test:  ## Run unit tests (no heavy deps needed)
	pytest -q

lint:  ## Lint with ruff
	ruff check .

smoke: sample ingest embed index  ## Run the full pipeline on the sample set
	@echo "Smoke pipeline complete — try: make ui"

clean:  ## Remove generated artifacts
	rm -f data/embeddings.npy data/ids.npy data/index.faiss* data/index.meta.json \
		data/index.ids.npy data/index.vectors.npy data/manifest.parquet eval/report.json
