# 🔍 Multimodal Image Search

> Search a large photo collection in plain language — type *"red car parked near a building at night"* and get the matching images back. Built with **CLIP** embeddings, a **FAISS** vector index, an optional **LLM** query layer, and a **FastAPI + Gradio** interface.

<p>
  <a href="https://github.com/stellasdeutsch-dev/multimodal-image-search/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/stellasdeutsch-dev/multimodal-image-search/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-portfolio%20project-orange">
</p>

This is an end-to-end system, not a notebook: a reproducible data pipeline, batch GPU embedding, an approximate-nearest-neighbor index, a serving API, a web UI, and a quantitative evaluation harness. Heavy compute (embedding, optional fine-tuning) runs on a GPU cluster via **SLURM**; everything else runs locally.

📄 Full design & phased build plan: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**

---

## ✨ Features

- **Text → image search** — natural-language queries over the whole collection.
- **Image → image search** — upload a photo, find visually similar ones.
- **Optional LLM layer** — rewrites vague queries into concrete, CLIP-friendly phrases (Anthropic Claude).
- **Scales** — batch embedding on GPU + FAISS `IVF`/`HNSW` indexing for fast search over 1M+ vectors.
- **Runs anywhere** — FAISS-optional: a NumPy brute-force fallback keeps the retrieval path (and tests) working with no heavy deps. A FAISS-built index also persists a vectors sidecar, so it **round-trips** into a NumPy-only environment instead of erroring at load time.
- **Enforced data contracts** — `src/schema.py` validates the manifest and embedding/id alignment at every stage boundary, so a malformed artifact fails fast with a clear message (not a downstream `KeyError`).
- **Measured** — Recall@K, mAP, and latency reported by an evaluation script.

---

## 🏗️ Architecture

```
                          ┌──────────────────────────────────────┐
                          │            Web UI (Gradio)            │
                          │   text box · image upload · gallery   │
                          └───────────────────┬──────────────────┘
                                              │ HTTP (JSON)
                          ┌───────────────────▼──────────────────┐
                          │           FastAPI backend             │
                          │  /search/text   /search/image         │
                          │  LLM rewrite · response caching        │
                          └───────┬───────────────────┬──────────┘
                                  │ query vector      │ ids → paths
                  ┌───────────────▼──────┐   ┌────────▼─────────────┐
                  │  CLIP text/image enc │   │  Vector index (FAISS) │
                  │  (OpenCLIP)          │   │  flat / IVF / HNSW     │
                  └──────────────────────┘   └────────┬─────────────┘
                                                       │
                                              ┌────────▼─────────────┐
                                              │   manifest.parquet    │
                                              │  id · path · w · h     │
                                              └──────────────────────┘

  OFFLINE (batch, GPU):  images → ingest/validate/dedup → CLIP encode → embeddings.npy → build index
```

**Online:** query → (optional LLM rewrite) → CLIP encode → ANN search → fetch paths from manifest → ranked results.
**Offline:** ingest images → batch CLIP encode on GPU → build the vector index.

---

## 🧰 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | **OpenCLIP** (`ViT-B-32`, swap to `ViT-L-14`) | Open weights, shared image/text space, easy batching. |
| Vector index | **FAISS** (flat → IVF/HNSW) + NumPy fallback | Fast in-process ANN; fallback keeps it runnable everywhere. |
| Backend | **FastAPI** + Uvicorn | Async serving, automatic OpenAPI docs. |
| LLM | **Anthropic Claude** (Haiku for query rewrite) | Reliable, cheap, optional. |
| Web UI | **Gradio** | Polished demo in minutes. |
| Tracking | **Weights & Biases** | Logs embedding throughput / experiments. |
| Tooling | `uv`/`pip`, `ruff`, `pytest`, YAML configs | Reproducible, config-driven. |

---

## 🚀 Quickstart (local toy demo — no GPU, no dataset)

> Use Python **3.10–3.12** (CLIP/FAISS wheels). On a Mac the tiny demo runs on CPU/MPS.

```bash
git clone <your-repo-url> && cd multimodal-image-search
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: pip install -r requirements.txt

make sample                      # generate 24 synthetic labeled images
make ingest                      # build the manifest
make embed                       # CLIP-encode the sample (CPU/MPS is fine here)
make index                       # build the vector index
make ui                          # open the Gradio UI at http://localhost:7860
```

Or query through the API:

```bash
make serve                       # http://localhost:8000/docs
curl -s -X POST localhost:8000/search/text \
     -H 'content-type: application/json' \
     -d '{"query":"a red circle","k":5}' | python -m json.tool
```

Run the tests (no heavy deps required):

```bash
make test
```

---

## 🖥️ Full pipeline on the GPU cluster (real data)

Heavy embedding runs on the university GPU via SLURM (see [PROJECT_PLAN.md](PROJECT_PLAN.md) for cluster specifics — server choice, NFS layout, staging to fast local disk).

```bash
# on the cluster (conda env "mmsearch" with the requirements installed)
python -m src.ingest  --config configs/data.yaml      # point images_dir at your dataset
sbatch slurm/embed.slurm                               # batch CLIP embedding on GPU
python -m src.index   --config configs/index.yaml --build

# recall/latency knobs: set at build in configs/index.yaml (index_type, ivf_nprobe,
# hnsw_ef_search), or override at serve/eval time in configs/serve.yaml.
# Use your own labeled query set for real data (see eval/queries.example.jsonl for the format).
python -m eval.evaluate --config configs/serve.yaml --queries eval/queries.example.jsonl
```

Scale up by switching `index_type` to `ivf` or `hnsw` in `configs/index.yaml` and pointing `configs/data.yaml:images_dir` at a larger set (e.g. COCO / Open Images / a LAION subset).

---

## 📁 Repo structure

```
configs/      data / embed / index / serve YAML configs
src/          ingest · embed · index · search · llm · api · model (CLIP)
web/          Gradio app
slurm/        hello + embed SLURM jobs (GPU)
eval/         Recall@K / mAP / latency harness + example queries
tests/        unit tests (NumPy fallback — run with no GPU)
scripts/      synthetic sample-image generator
PROJECT_PLAN.md   full phased plan & design
```

---

## 📊 Results

Fill in after running `eval/evaluate.py` on a labeled query set:

| Dataset | # images | Index | Recall@1 | Recall@5 | Recall@10 | mAP | p95 latency | Embed throughput |
|---|---|---|---|---|---|---|---|---|
| _e.g. COCO val_ | _—_ | _HNSW_ | _—_ | _—_ | _—_ | _—_ | _— ms_ | _— img/s_ |

---

## 🗺️ Roadmap

- [ ] Hybrid text + image-to-image search
- [ ] Domain fine-tuning of CLIP (report the Recall@K lift)
- [ ] Metadata filtering (move to Qdrant)
- [ ] ONNX/TensorRT export of the encoder
- [ ] Cross-encoder re-ranking stage

---

## 🧩 Extending this

The architecture is built around named seams, so each piece swaps without touching the rest:

| Want to… | Change | Why it's isolated |
|---|---|---|
| Swap the embedding model | `src/model.py` `ClipEncoder`, or pass any object satisfying the `Encoder` protocol (`src/search.py`) to `Searcher` | `Searcher` depends on the `Encoder` protocol, not on CLIP — see `tests/test_api.py` / `test_search.py` for a 6-line stub encoder. |
| Add an index type | `VectorIndex._build_faiss` in `src/index.py` (`flat`/`ivf`/`hnsw` → add a branch) | The `search()`/`save()`/`load()` contract is backend-agnostic. |
| Use a different vector DB (Qdrant…) | implement the `VectorIndex` interface (`build`/`search`/`save`/`load`) | `Searcher` only calls those four methods. |
| Change the LLM layer | `src/llm.py` `QueryAssistant` | Disabled by default; the API/UI call `rewrite()` which is a pass-through when off. |
| Add a manifest field | extend `MANIFEST_COLUMNS` in `src/schema.py` | One source of truth, validated at every boundary. |

## 📜 License

MIT — see [LICENSE](LICENSE).
