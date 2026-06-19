# CLAUDE.md — build guide for this repo

This repo implements **Multimodal Image Search**. The full phased plan lives in
[PROJECT_PLAN.md](PROJECT_PLAN.md) — read it before making large changes.

## Conventions
- Python ≥3.10. Code in `src/`, configs in `configs/` (YAML), tests in `tests/`.
- Each pipeline stage is a module with a CLI:
  `python -m src.<stage> --config configs/<stage>.yaml`.
- Embeddings are **L2-normalized**; similarity = inner product (== cosine).
- Keep changes **config-driven** — don't hardcode paths, models, or hyperparameters.
- `src/index.py` and `src/search.py` must keep working **without** torch/faiss
  (NumPy fallback) so tests run anywhere.

## Where things run
- **Heavy steps (embedding, fine-tuning) run on the uni GPU via SLURM** (`slurm/`),
  never on the Mac. Local = code + smoke tests on a tiny sample only.
- Loop: edit on Mac → `pytest -q` → push → pull on GPU → `sbatch slurm/embed.slurm`
  → monitor (`squeue`, W&B) → copy results back → analyze locally.

## Don't commit
data, embeddings, indexes, `.env` — see `.gitignore`.
