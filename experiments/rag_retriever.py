"""Retrieval-Augmented Few-Shot (RAG-FS) example retriever.

Embeds a training pool once with sentence-transformers, then at inference time
retrieves the top-k most semantically similar examples for each test datum.
All embeddings are L2-normalised so cosine similarity reduces to a dot product.

Typical usage (weather):
    retriever = RAGRetriever()
    retriever.fit(train_data)                    # embed once (slow, do outside loop)
    examples, scores = retriever.retrieve(test_datum, k=4)
    prompt = format_ragfs_weather_prompt(examples)

Typical usage (extreme, inside k-fold):
    retriever = RAGRetriever()
    retriever.fit(train_pool)
    for datum in test_data:
        examples, _ = retriever.retrieve(datum, k=n_shots)
        fewshot = format_ragfs_extreme_prompt(examples)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class RAGRetriever:
    """Embed a pool of training dicts, then retrieve similar examples at query time.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model name.  'all-MiniLM-L6-v2' is fast and
        compact (22 M params); 'all-mpnet-base-v2' is slightly better quality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._sbert = SentenceTransformer(model_name)
        self._pool: List[dict] = []
        self._embeddings: np.ndarray | None = None
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, pool: List[dict], text_key: str = "observation") -> None:
        """Embed all training examples and cache their embeddings.

        Parameters
        ----------
        pool     : list of training dicts (each must have key ``text_key``)
        text_key : field to embed (default "observation")
        """
        self._pool = list(pool)
        texts = [str(d.get(text_key, "")) for d in pool]

        print(f"[RAGRetriever] Embedding {len(texts)} training examples "
              f"with '{self._model_name}' ...")
        self._embeddings = self._sbert.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        print(f"[RAGRetriever] Done — embedding shape: {self._embeddings.shape}")

    def retrieve(
        self,
        datum: dict,
        k: int = 4,
        text_key: str = "observation",
    ) -> Tuple[List[dict], List[float]]:
        """Return top-k training examples most similar to ``datum``.

        Parameters
        ----------
        datum    : test datum dict
        k        : number of examples to retrieve
        text_key : field used for similarity (must match the one used in fit)

        Returns
        -------
        (examples, similarities)
            Both sorted descending by cosine similarity.
            Returns ([], []) if the retriever has not been fitted yet.
        """
        if self._embeddings is None or not self._pool:
            return [], []

        query_text = str(datum.get(text_key, ""))
        query_emb = self._sbert.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # cosine similarity = dot product because embeddings are normalised
        sims: np.ndarray = (self._embeddings @ query_emb.T).flatten()

        k_eff = min(k, len(self._pool))
        top_idx = np.argsort(sims)[::-1][:k_eff]

        return (
            [self._pool[i] for i in top_idx],
            sims[top_idx].tolist(),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def is_fitted(self) -> bool:
        return self._embeddings is not None and len(self._pool) > 0
