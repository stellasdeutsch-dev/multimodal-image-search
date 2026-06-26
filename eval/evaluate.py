"""Phase 7 — retrieval evaluation: Recall@K, mAP, and query latency.

Queries file (JSONL), one object per line:
    {"query": "a red circle", "relevant_ids": [3, 11]}

Usage:  python -m eval.evaluate --config configs/serve.yaml --queries eval/queries.example.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from src.common import load_config, setup_logging
from src.search import Searcher

log = logging.getLogger("mmsearch")


def load_queries(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def recall_at_k(ranked_ids: list[int], relevant: list[int], k: int) -> float:
    return 1.0 if set(relevant) & set(ranked_ids[:k]) else 0.0


def average_precision(ranked_ids: list[int], relevant: list[int]) -> float:
    rel = set(relevant)
    hits, ap = 0, 0.0
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in rel:
            hits += 1
            ap += hits / rank
    return ap / max(1, len(rel))


def evaluate(cfg: dict, queries_path: str, ks=(1, 5, 10)) -> dict:
    searcher = Searcher.from_config(cfg)
    items = load_queries(queries_path)
    recalls = {k: [] for k in ks}
    aps, latencies = [], []
    for it in items:
        t0 = time.time()
        results = searcher.search_text(it["query"], max(ks))
        latencies.append((time.time() - t0) * 1000.0)
        ranked = [r["id"] for r in results]
        rel = it["relevant_ids"]
        for k in ks:
            recalls[k].append(recall_at_k(ranked, rel, k))
        aps.append(average_precision(ranked, rel))

    report = {"n_queries": len(items)}
    if not items:                       # avoid np.mean([]) -> NaN (invalid JSON)
        for k in ks:
            report[f"recall@{k}"] = 0.0
        report.update({"mAP": 0.0, "latency_ms_p50": 0.0, "latency_ms_p95": 0.0})
        return report
    for k in ks:
        report[f"recall@{k}"] = round(float(np.mean(recalls[k])), 4)
    report["mAP"] = round(float(np.mean(aps)), 4)
    report["latency_ms_p50"] = round(float(np.percentile(latencies, 50)), 2)
    report["latency_ms_p95"] = round(float(np.percentile(latencies, 95)), 2)
    return report


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("--config", default="configs/serve.yaml")
    ap.add_argument("--queries", default="eval/queries.example.jsonl")
    ap.add_argument("--out", default="eval/report.json")
    args = ap.parse_args()
    report = evaluate(load_config(args.config), args.queries)
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
