"""Hybrid retrieval: dense cosine similarity + BM25, fused with Reciprocal Rank Fusion."""
import re
from dataclasses import dataclass, replace

import numpy as np
from rank_bm25 import BM25Okapi

from store import ChunkSnapshot

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    doc_id: int
    page: int
    text: str
    score: float


class HybridIndex:
    RRF_K = 60        # standard RRF constant
    CANDIDATES = 50   # candidates taken from each ranker before fusion

    def __init__(self, snapshot: ChunkSnapshot):
        self.s = snapshot
        self._bm25 = BM25Okapi([tokenize(t) for t in snapshot.texts]) if snapshot.texts else None

    def __len__(self) -> int:
        return len(self.s.texts)

    def search(self, query_vec, query_text, top_k, doc_ids: set[int] | None = None) -> list[Hit]:
        if self._bm25 is None:
            return []
        mask = np.isin(self.s.doc_ids, list(doc_ids)) if doc_ids else np.ones(len(self), dtype=bool)
        if not mask.any():
            return []

        dense = self.s.embeddings @ query_vec
        sparse = np.asarray(self._bm25.get_scores(tokenize(query_text)))

        fused = np.zeros(len(self))
        # BM25 only contributes chunks that actually share terms with the query.
        for scores, valid in ((dense, mask), (sparse, mask & (sparse > 0))):
            n = min(self.CANDIDATES, int(valid.sum()))
            if n == 0:
                continue
            masked = np.where(valid, scores, -np.inf)
            top = np.argpartition(-masked, n - 1)[:n]
            top = top[np.argsort(-masked[top])]
            fused[top] += 1.0 / (self.RRF_K + np.arange(1, n + 1))

        best = np.argsort(-fused)[:top_k]
        return [
            Hit(int(self.s.doc_ids[i]), int(self.s.pages[i]), self.s.texts[i], float(fused[i]))
            for i in best
            if fused[i] > 0
        ]


def rerank(reranker, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if not hits:
        return hits
    scores = reranker.scores(query, [h.text for h in hits])
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)[:top_k]
    return [replace(h, score=s) for s, h in ranked]
