"""Phase 5 — optional LLM layer (Anthropic Claude).

Rewrites vague queries into concrete, CLIP-friendly phrases and can explain why
a result matched. Fully optional: if disabled or no API key, rewrite() is a
pass-through so the rest of the system works unchanged."""

from __future__ import annotations

import logging
import os

log = logging.getLogger("mmsearch")

REWRITE_SYSTEM = (
    "You rewrite a user's image-search query into a single, concrete, visually "
    "descriptive phrase optimized for CLIP retrieval. Keep it under 25 words. "
    "Return only the rewritten query, with no preamble or quotes."
)


class QueryAssistant:
    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.model = cfg.get("model", "claude-haiku-4-5-20251001")
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self._client = None
        if self.enabled:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                log.warning("LLM enabled but ANTHROPIC_API_KEY is unset; disabling.")
                self.enabled = False
            else:
                try:
                    import anthropic

                    self._client = anthropic.Anthropic()
                except Exception as e:  # pragma: no cover
                    log.warning("anthropic unavailable (%s); disabling LLM.", e)
                    self.enabled = False

    def rewrite(self, query: str) -> str:
        if not self.enabled:
            return query
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REWRITE_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            return resp.content[0].text.strip() or query
        except Exception as e:  # pragma: no cover
            log.warning("rewrite failed (%s); using original query.", e)
            return query

    def explain(self, query: str, result: dict) -> str:
        if not self.enabled:
            return ""
        prompt = (
            f"User searched: {query!r}. A retrieved image is {result.get('path')!r} "
            f"with similarity {result.get('score', 0):.3f}. In one short sentence, "
            "explain why it might match."
        )
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:  # pragma: no cover
            log.warning("explain failed (%s)", e)
            return ""
