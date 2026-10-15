"""Local embeddings and re-ranking via fastembed (ONNX runtime, no PyTorch needed)."""
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        # Explicit cache_dir: fastembed's default lives in the OS temp folder and can get wiped.
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)
        return _normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.asarray(list(self._model.query_embed(query)), dtype=np.float32)
        return _normalize(vec)[0]


class Reranker:
    """Cross-encoder that reads query and passage together; much sharper than vector similarity."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]
