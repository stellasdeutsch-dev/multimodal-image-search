"""Phase 6 — Gradio web UI. Runs the Searcher in-process.

Run:  python web/app.py     (set MMSEARCH_CONFIG to choose a serve config)
"""

from __future__ import annotations

import os
import sys

# Allow `python web/app.py` from the repo root to import the src package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402

from src.common import load_config  # noqa: E402
from src.search import Searcher  # noqa: E402

CFG_PATH = os.environ.get("MMSEARCH_CONFIG", "configs/serve.yaml")
cfg = load_config(CFG_PATH)
searcher = Searcher.from_config(cfg)
DEFAULT_K = cfg.get("top_k", 24)


def _gallery(results: list[dict]):
    return [(r["path"], f"{r['score']:.3f}") for r in results]


def text_search(query: str, k: int):
    if not query:
        return []
    return _gallery(searcher.search_text(query, int(k)))


def image_search(image, k: int):
    if image is None:
        return []
    return _gallery(searcher.search_image(image, int(k)))


with gr.Blocks(title="Multimodal Image Search") as demo:
    gr.Markdown("# 🔍 Multimodal Image Search\nSearch a photo collection by text or by image (CLIP embeddings).")

    with gr.Tab("Text → Image"):
        query = gr.Textbox(label="Query", placeholder="e.g. red car parked near a building at night")
        k1 = gr.Slider(1, 50, value=DEFAULT_K, step=1, label="Results")
        btn1 = gr.Button("Search", variant="primary")
        gallery1 = gr.Gallery(label="Results", columns=4, height=600)
        btn1.click(text_search, [query, k1], gallery1)
        query.submit(text_search, [query, k1], gallery1)

    with gr.Tab("Image → Image"):
        image = gr.Image(type="pil", label="Upload an image")
        k2 = gr.Slider(1, 50, value=DEFAULT_K, step=1, label="Results")
        btn2 = gr.Button("Search", variant="primary")
        gallery2 = gr.Gallery(label="Results", columns=4, height=600)
        btn2.click(image_search, [image, k2], gallery2)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
