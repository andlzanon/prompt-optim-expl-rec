from __future__ import annotations

from src.representation.base_representation import BaseRepresentation

from typing import Any, Optional, Sequence
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class SBERTRepresentation(BaseRepresentation):
    """
    Representation backend based on a Sentence-Transformers model.

    This class adapts ``SentenceTransformer`` to the project's
    ``BaseRepresentation`` interface. It loads the configured SBERT model
    during initialization and exposes an ``encode`` method that returns NumPy
    embeddings for input texts.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        device: Optional[str] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the SBERT representation backend and load the model.

        Parameters
        ----------
        model_name : str, default="all-MiniLM-L6-v2"
            Sentence-Transformers model identifier used to load the encoder.
        normalize : bool, default=True
            Whether the underlying ``SentenceTransformer.encode`` call should
            return normalized embeddings.
        device : Optional[str], default=None
            Device used to run the model. When ``None``, the class selects
            ``"cuda"`` if available, otherwise ``"cpu"``.
        encode_kwargs : Optional[dict[str, Any]], default=None
            Additional keyword arguments forwarded to
            ``SentenceTransformer.encode``. The default configuration always
            includes ``show_progress_bar=False`` unless explicitly overridden.

        Returns
        -------
        None
            This constructor initializes the instance in place.

        Raises
        ------
        Exception
            Any exception raised by ``SentenceTransformer`` model loading may
            propagate.
        """

        self.model_name = model_name
        self.normalize = bool(normalize)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encode_kwargs = {"show_progress_bar": False, **(encode_kwargs or {})}
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a sequence of texts into a NumPy embedding matrix.

        The method converts each input item into a string, replacing ``None``
        values with empty strings, and delegates the embedding computation to
        the loaded ``SentenceTransformer`` model. The returned embeddings are
        always converted to ``np.float32``.

        Parameters
        ----------
        texts : Sequence[str]
            Sequence of input texts to encode. Individual elements may be
            ``None``, in which case they are converted to empty strings before
            encoding.

        Returns
        -------
        np.ndarray
            Two-dimensional NumPy array of embeddings with dtype
            ``np.float32``. When ``texts`` is empty, the method returns an
            empty array with shape ``(0, self.embedding_dimension)``.

        Raises
        ------
        Exception
            Any exception raised by the underlying model's ``encode`` method
            may propagate.

        Notes
        -----
        This method is the concrete implementation of the abstract
        ``BaseRepresentation.encode`` interface for SBERT-based embeddings.
        """

        # Convert all inputs into strings so the encoder receives a stable type.
        prepared_texts = ["" if text is None else str(text) for text in texts]
        if not prepared_texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = self.model.encode(
            prepared_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            **self.encode_kwargs,
        )
        return np.asarray(embeddings, dtype=np.float32)