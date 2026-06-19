# Runbook — running Multimodal Image Search on the ITEC GPU cluster

Step-by-step to take this from scaffold to **real metrics in the README**.
Code/tests/UI on the Mac; batch embedding on the GPU.

## 0. One-time setup (on the cluster)
```bash
ssh -p 2022 USER@gpu6.itec.aau.at            # VPN required; gpu6 = 2× RTX 6000 Ada
conda create -n mmsearch python=3.11 -y && conda activate mmsearch
git clone https://github.com/stellasdeutsch-dev/multimodal-image-search ~/mmsearch
cd ~/mmsearch
pip install -r requirements.txt              # torch from the CUDA index if needed
mkdir -p logs
sbatch slurm/hello.slurm && squeue -u $USER  # Phase 0: confirm GPU is visible
```

## 1. Get a dataset
- **Unsplash Lite** (~25k, easy start), **COCO/Open Images** (mid), or a **LAION** subset (scale) — from `/shares/datasets` if present, else download into `data/images/`.
- For speed, stage onto NVMe: `rsync -a /shares/datasets/<set>/ /fastlocal/$USER/images/` and point `configs/data.yaml:images_dir` there (NVMe is **not** backed up — copy embeddings/index back to `~`).

## 2. Build the index (GPU for embedding)
```bash
python -m src.ingest --config configs/data.yaml     # manifest.parquet
sbatch slurm/embed.slurm                             # batch CLIP embedding -> embeddings.npy
squeue -u $USER                                      # watch; logs/embed_<jobid>.out
python -m src.index  --config configs/index.yaml --build
```
Tune `configs/index.yaml` (`index_type: ivf|hnsw`, `ivf_nprobe`, `hnsw_ef_search`) for the recall/latency trade-off at scale.

## 3. Evaluate
```bash
# build a small labeled query set (query -> relevant image ids) like eval/queries.example.jsonl, then:
python -m eval.evaluate --config configs/serve.yaml --queries eval/queries.jsonl
```
Records Recall@1/5/10, mAP, and latency p50/p95.

## 4. Bring results back & serve locally
```bash
# from the Mac:
rsync -avP -e "ssh -p 2022" USER@gpu6.itec.aau.at:~/mmsearch/data ./    # manifest + index + embeddings
make serve   # FastAPI at :8000/docs
make ui      # Gradio at :7860
```

## 5. Fill the README
- Add the eval numbers to the **Results** table: Recall@K, mAP, latency p50/p95, and the embedding **throughput** (img/s) logged during Phase 2.
- Record a short demo GIF of a text search and an image-to-image search.
- Write the **error-analysis** note (see `notebooks/error_analysis.ipynb`): which queries fail and why (domain gap, embedding blind spots, ambiguous text), and what you changed.
