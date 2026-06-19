# Multimodal Image Search (CLIP + LLM + Web UI)

> A natural-language image search engine: type "red car parked near a building at night" and get the matching images back from a collection of hundreds of thousands of photos — powered by CLIP embeddings, a vector database, and an LLM query layer, served through a web UI.

This document is **both** the build brief (hand each phase to Claude Code) **and** the portfolio writeup (architecture, skills, metrics) for the project.

---

## 1. What it does

- **Text → image search.** User types a free-form query; the system returns the top-K most relevant images, ranked, with similarity scores.
- **Image → image search.** Upload a photo, get visually similar images.
- **LLM query layer.** A language model rewrites vague queries into precise ones, can re-rank results, and can explain *why* a result matched.
- **Scale.** Designed to index 25k → 1M+ images, with embeddings generated in batch on a GPU and served from an approximate-nearest-neighbor (ANN) index in milliseconds.

---

## 2. Why this is a strong AI / Data Engineer project (CV signal)

This project deliberately exercises the skills hiring managers look for — not just "trained a model," but **built a system**:

- **Batch embedding at scale** — encoding a large image corpus on GPU, with throughput (images/sec) tuning, `fp16`, and batch-size optimization.
- **Vector indexing & ANN tuning** — FAISS/Qdrant, IVF/HNSW, the recall-vs-latency trade-off (this is the real engineering challenge).
- **Retrieval evaluation** — Recall@K, mAP, latency percentiles — quantitative, not vibes.
- **LLM integration** — query understanding, RAG-style re-ranking and explanations.
- **End-to-end serving** — FastAPI backend, async, caching, a usable web UI.
- **Reproducible pipelines** — config-driven scripts (not one giant notebook), versioned data manifests.
- **Experiment tracking** — Weights & Biases runs for every GPU job.
- **Hybrid local/cluster workflow** — develop on a Mac, train/embed on a SLURM GPU cluster (real-world MLOps muscle).

---

## 3. Architecture

```
                          ┌──────────────────────────────────────┐
                          │              Web UI                   │
                          │     (Gradio  →  or React later)       │
                          │  text box · upload · results grid     │
                          └───────────────────┬──────────────────┘
                                              │ HTTP (JSON)
                          ┌───────────────────▼──────────────────┐
                          │           FastAPI backend             │
                          │  /search/text   /search/image         │
                          │  async · response caching             │
                          └───────┬───────────────────┬──────────┘
                                  │                   │
                  ┌───────────────▼──────┐   ┌────────▼─────────────┐
                  │     LLM layer        │   │   Query encoder      │
                  │ rewrite · re-rank ·  │   │  CLIP/SigLIP text    │
                  │ "why it matched"     │   │  → query vector      │
                  └───────────────┬──────┘   └────────┬─────────────┘
                                  │                   │ vector
                                  │          ┌────────▼─────────────┐
                                  └─────────▶│   Vector index       │
                                             │  FAISS (IVF/HNSW)    │
                                             │  → top-K image IDs   │
                                             └────────┬─────────────┘
                                                      │ IDs → paths/meta
                                             ┌────────▼─────────────┐
                                             │   manifest.parquet   │
                                             │  id · path · meta    │
                                             └──────────────────────┘

         ── OFFLINE (batch, on GPU) ───────────────────────────────────────
         images ─▶ ingest/validate/dedup ─▶ CLIP image encoder ─▶ embeddings.npy ─▶ build FAISS index
```

**Online path:** query → (LLM rewrite) → text encoder → vector → ANN search → IDs → fetch paths/metadata from the manifest → (LLM re-rank/explain) → return.
**Offline path:** ingest images → CLIP image encoder (batch, GPU) → embeddings → build index.

---

## 4. Tech stack (chosen, with rationale)

| Layer | Choice | Why |
|---|---|---|
| Embeddings | **OpenCLIP** (`ViT-B/32` to start, `ViT-L/14` for quality) | Open weights, many checkpoints, easy batch encoding. SigLIP is a strong drop-in alternative for better retrieval. |
| Vector index | **FAISS** (start flat → IVF/HNSW) | Fastest path, runs in-process, no server to manage. *Fork:* use **Qdrant** if you want a richer "real database" story (filtering, persistence, REST). |
| Backend | **FastAPI** | Async, fast, automatic OpenAPI docs, the standard for ML serving. |
| LLM | **Hosted API** (Anthropic Claude) | Reliable, zero infra. *Fork:* run a small local model on the uni GPU if you want a "self-hosted LLM" line on the CV. |
| Web UI | **Gradio** (Phase 6) | Polished demo in hours. *Fork:* rebuild in **React** if you want a stronger frontend signal. |
| Experiment tracking | **Weights & Biases** | Cloud-based, works from Mac and cluster, no self-hosting (simpler than MLflow here). |
| Env / tooling | **uv** or **miniforge**, `ruff`, `pre-commit`, **Hydra**/YAML configs | Reproducible, fast, config-driven. |
| Data format | **Parquet** (manifest), **`.npy`/`.fvecs`** (embeddings) | Columnar metadata + compact vector storage. |

---

## 5. Repo structure

```
multimodal-image-search/
├── CLAUDE.md                  # short brief for Claude Code (points here)
├── PROJECT_PLAN.md            # this file
├── README.md                  # CV-facing front page (written in Phase 8)
├── Makefile                   # make setup / embed / index / serve
├── pyproject.toml             # deps (uv/pip), ruff config
├── configs/
│   ├── data.yaml              # dataset paths, image size, dedup
│   ├── embed.yaml             # model, batch size, fp16, device
│   ├── index.yaml             # FAISS type, nlist/nprobe / HNSW M/efSearch
│   └── serve.yaml             # API + LLM settings
├── src/
│   ├── ingest.py              # scan images → manifest.parquet, validate, dedup
│   ├── embed.py               # batch CLIP encoding → embeddings (GPU entrypoint)
│   ├── model.py               # CLIP encoder wrapper (OpenCLIP)
│   ├── index.py               # build/load FAISS index, search()
│   ├── search.py              # query → encode → search → results (core logic)
│   ├── llm.py                 # query rewrite, re-rank, explanations
│   └── api.py                 # FastAPI app
├── web/                       # Gradio app (or React app later)
├── slurm/
│   ├── embed.slurm            # GPU batch embedding job
│   └── hello.slurm            # connectivity smoke test
├── eval/
│   ├── queries.example.jsonl  # labeled eval queries (example format)
│   └── evaluate.py            # Recall@K, mAP, latency
├── notebooks/                 # error analysis, ad-hoc exploration
└── logs/                      # SLURM + run logs (gitignored)
```

---

## 6. Infrastructure & workflow (Mac M3 + ITEC GPU cluster)

### The development loop (every GPU phase)
1. **Write & smoke-test on the Mac M3** — code, configs, run on a tiny subset (MPS is fine for sanity checks; **never for real training/embedding**).
2. **`git push`** from Mac → **`git pull`** on the GPU server.
3. **`sbatch slurm/<job>.slurm`** to launch the real run.
4. **Monitor** via `squeue -u $USER` + the **W&B** dashboard (logs throughput, loss, metrics live).
5. **Copy results back** to the Mac → analyze locally (notebooks, plots).

### ITEC / AAU cluster specifics
- **Connect (VPN required):** `ssh -p 2022 USER@gpu6.itec.aau.at`
- **Server choice:**
  - **`gpu6`** — 2× RTX 6000 Ada → **fastest, use this by default**.
  - **`gpu4`** — 2× RTX 8000, 48 GB → use when you need the larger VRAM.
  - `gpu3` — locked until **2026-06-21**; `gpu5` — locked until **2026-06-29** (avoid until freed).
- **Environment:** install **miniforge in your NFS home (`~`)** — shared across login servers and backed up. Code lives in `~`, kept in sync via `git pull`.
- **Datasets:** read from **`/shares/datasets`** (10 TB NFS). For training speed, stage the working set to **`/fastlocal/$USER`** (NVMe) — **not backed up**, so copy any checkpoints/embeddings you want to keep back to `~`.
- **All GPU work goes through SLURM** (`sbatch`) — no interactive GPU hogging.
- *Optional:* email `support@itec.aau.at` to enable SSH TCP forwarding → use VS Code Remote / Jupyter on the GPU box during development.

---

## 7. SLURM job template

`slurm/embed.slurm` — confirm partition/`gres` names first with `sinfo`:

```bash
#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --gres=gpu:1            # gpu:2 to use both GPUs on gpu6/gpu4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mmsearch

srun python -m src.embed --config configs/embed.yaml --fp16
```

`slurm/hello.slurm` (Phase 0 connectivity check) does the same but runs `python -c "import torch; print(torch.cuda.get_device_name(0))"`.

---

## 8. Dataset options (pick by ambition)

| Tier | Dataset | Size | Use it for |
|---|---|---|---|
| Easy start | **Unsplash Lite** | ~25k images | Get the full pipeline working end-to-end fast. |
| Mid | **COCO** / **Open Images** (subset) | 100k–1M | Realistic scale, has captions/labels → easy eval set. |
| Scale | **LAION** subset | 1M+ | Show off indexing at scale (the impressive part). |

Start at **Unsplash Lite**, prove the whole loop, then scale the *same code* up to a larger set on `/shares/datasets`.

---

## 9. Phased execution plan

Each phase is a discrete task you can hand to Claude Code. "Where" = local (Mac) vs GPU (cluster).

### Phase 0 — Scaffold · *local*
- Init repo, env (`uv`/miniforge), `ruff` + `pre-commit`, configs, `Makefile`, W&B init.
- Set up SSH to `gpu6`, install miniforge in `~`, submit `slurm/hello.slurm`.
- **Done when:** `make setup` works locally **and** the hello-world GPU job prints the GPU name on the cluster.

### Phase 1 — Data pipeline · *local → cluster*
- `src/ingest.py`: scan images → `manifest.parquet` (id, path, width, height); validate (drop corrupt), dedup (content hash). Resize/normalize happens later at embed time via the CLIP preprocess transform.
- **Done when:** `manifest.parquet` covers N images on the server, validated and deduped.

### Phase 2 — Embedding generation · *GPU*
- `src/embed.py`: batch-encode all images with OpenCLIP; write `embeddings.npy` (+ id alignment) to `/fastlocal`, copy back to `~`.
- Tune batch size + `fp16`; log **images/sec** to W&B.
- **Done when:** the full corpus is embedded, with a throughput report (images/sec, GPU utilization).

### Phase 3 — Vector index · *local/GPU*
- `src/index.py`: build FAISS index (flat → IVF or HNSW). Tune `nprobe` / `efSearch`.
- Benchmark **Recall@K vs latency**.
- **Done when:** `search()` returns relevant hits and you have a recall-vs-latency curve.

### Phase 4 — Search API · *local*
- `src/api.py`: FastAPI with `/search/text` (text→embed→search) and `/search/image` (upload→embed→search); ranked results with scores; async + response caching.
- **Done when:** the API serves both query types with scored, ranked results.

### Phase 5 — LLM layer · *local (hosted) or GPU (local model)*
- `src/llm.py`: query rewriting/expansion (vague → precise), optional re-ranking, natural-language "why this matched" explanations.
- **Done when:** vague queries measurably improve (show a before/after example).

### Phase 6 — Web UI · *local*
- Gradio app: search bar, image upload, results grid with scores, filters. (React if you want stronger frontend signal.)
- **Done when:** a person can use it as a live demo end-to-end.

### Phase 7 — Evaluation & error analysis · *local*
- `eval/evaluate.py`: Recall@K, mAP on a labeled query set. Catalog failure cases (domain gap, embedding blind spots, ambiguous queries) with reasons.
- *Stretch:* fine-tune CLIP on a domain subset and show the lift.
- **Done when:** you have an eval report + a written error-analysis section.

### Phase 8 — Ship · *local*
- `README.md` with architecture diagram, demo GIF, metrics table. Dockerize (build x86 image for deployment). Short writeup/blog.
- **Done when:** repo is portfolio-ready and a stranger can run it from the README.

---

## 10. Evaluation & metrics to report

- **Retrieval quality:** Recall@1 / @5 / @10, mAP on the labeled query set.
- **Latency:** p50 / p95 query latency (ms).
- **Throughput:** embedding images/sec (with GPU + batch + `fp16` config).
- **Index:** build time, memory footprint, recall-vs-`nprobe`/`efSearch` curve.
- **LLM layer:** before/after retrieval quality on vague queries.

Put these in a table in the README — quantified results are what make the project credible.

---

## 11. Stretch goals

- **Hybrid search** — combine text and image-to-image queries.
- **Domain fine-tuning** — fine-tune CLIP on your data and show the Recall@K lift.
- **Faster serving** — ONNX/TensorRT export of the encoder.
- **Better ranking** — a cross-encoder / multi-vector re-ranking stage.
- **Metadata filtering** — move to Qdrant for filtered search (e.g. "night photos only").

---

## 12. Deliverables checklist (for the CV)

- [ ] Public GitHub repo, clean README, reproducible `make` targets.
- [ ] Live or recorded **demo** (GIF/video of a real search).
- [ ] **Metrics table** (Recall@K, mAP, latency, throughput).
- [ ] **Architecture diagram** (the one above, cleaned up).
- [ ] **Error-analysis writeup** — what failed, why, what you changed.
- [ ] One-paragraph résumé bullet: *"Built an end-to-end multimodal image search engine (CLIP + FAISS + LLM + FastAPI/Gradio) indexing N images; achieved Recall@10 of X% at Y ms p95; batch-embedded the corpus at Z img/s on a SLURM GPU cluster."*

---

## 13. Quickstart commands

```bash
# Local setup (Mac M3)
make setup                         # env + deps + pre-commit + W&B login

# Phase 0 — verify cluster
ssh -p 2022 USER@gpu6.itec.aau.at
sbatch slurm/hello.slurm && squeue -u $USER

# Phase 1–2 — data + embeddings (on cluster)
python -m src.ingest  --config configs/data.yaml      # build manifest
sbatch slurm/embed.slurm                               # batch-encode on GPU

# Phase 3 — index (local or cluster)
python -m src.index   --config configs/index.yaml --build

# Phase 4 + 6 — serve API + UI (local)
uvicorn src.api:app --reload                           # backend
python web/app.py                                      # Gradio UI

# Phase 7 — evaluate
python -m eval.evaluate --config configs/serve.yaml --queries eval/queries.example.jsonl
```

---

*Build order: do Phase 0 → 6 to get a working demo fast, then 7 → 8 to make it portfolio-grade. Each phase is self-contained — hand it to Claude Code one at a time.*
