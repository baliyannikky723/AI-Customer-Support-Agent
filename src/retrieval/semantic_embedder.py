"""Semantic Embedding Engine using Sentence-Transformers.

Provides a reproducible, configurable wrapper around sentence transformer models
for dense vector representations of customer support queries and historical cases.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticEmbedder:
    """Configurable dense sentence embedder for semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        device: Union[str, None] = None,
    ):
        """Initialize SentenceTransformer embedding model.

        Args:
            model_name: HuggingFace model repo or local path.
            normalize_embeddings: If True, vectors are L2 normalized to unit length.
            batch_size: Batch size for inference encoding.
            device: Device to run model on (None auto-selects cuda/cpu).
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Union[int, None] = None,
        normalize_embeddings: Union[bool, None] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text or list of texts into dense numpy vector embeddings.

        Args:
            texts: Single string or list of text strings.
            batch_size: Optional override for batch size.
            normalize_embeddings: Optional override for L2 normalization.
            show_progress_bar: If True, display tqdm progress bar.

        Returns:
            np.ndarray of shape (N, dimension) as float32.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        bs = batch_size or self.batch_size
        norm = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=norm,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)
