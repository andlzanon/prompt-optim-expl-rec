from __future__ import annotations

from src.representation.base_representation import BaseRepresentation

from typing import Optional, Sequence
from llm2vec import LLM2Vec
import numpy as np
import torch

class LLM2VecRepresentation(BaseRepresentation):
    """
    Representation backend that encodes text with an LLM2Vec model.

    This class adapts the external ``LLM2Vec`` encoder to the project's
    ``BaseRepresentation`` interface. It loads the configured base and
    supervised checkpoint pair during initialization and exposes a single
    ``encode`` method that returns NumPy embeddings.
    """

    def __init__(
        self,
        model_name: str = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
        supervised_model_name: str = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised",
        pooling_mode: str = "mean",
        max_length: int = 512,
        normalize: bool = True,
        device_map: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Initialize the representation backend and load the underlying model.

        Parameters
        ----------
        model_name : str, default="McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
            Base LLM2Vec model checkpoint identifier passed to
            ``LLM2Vec.from_pretrained``.
        supervised_model_name : str, default="McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
            PEFT or supervised checkpoint identifier passed as
            ``peft_model_name_or_path``.
        pooling_mode : str, default="mean"
            Pooling strategy forwarded to the LLM2Vec loader.
        max_length : int, default=512
            Maximum tokenized sequence length forwarded to the model loader.
        normalize : bool, default=True
            Whether the output embeddings should be L2-normalized in
            ``encode``.
        device_map : Optional[str], default=None
            Device placement hint used by LLM2Vec. When ``None``, the class
            selects ``"cuda"`` if available, otherwise ``"cpu"``.
        torch_dtype : torch.dtype, default=torch.bfloat16
            Torch data type used when loading the model.

        Returns
        -------
        None
            This constructor initializes the instance in place.

        Raises
        ------
        RuntimeError
            Propagated from ``self._load_model()`` when the LLM2Vec model
            cannot be loaded with the expected local loader configuration.
        """

        self.model_name = model_name
        self.supervised_model_name = supervised_model_name
        self.pooling_mode = pooling_mode
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.device_map = device_map or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.model = self._load_model()
        self.embedding_dimension = int(self.model.model.config.hidden_size)

    def _load_model(self) -> LLM2Vec:
        """
        Load the configured LLM2Vec model with the project's expected settings.

        The method uses ``LLM2Vec.from_pretrained`` with the local loader path
        expected by this project, including the supervised checkpoint, pooling
        mode, maximum length, device mapping, and torch dtype.

        Parameters
        ----------
        None
            This helper uses instance attributes configured during
            initialization.

        Returns
        -------
        LLM2Vec
            Loaded LLM2Vec model instance ready to encode text.

        Raises
        ------
        RuntimeError
            Raised when model loading fails for any reason. The original
            exception is preserved as the cause.

        Notes
        -----
        This wrapper intentionally avoids relying on remote model code
        revisions that may require a newer ``transformers`` version than the
        one pinned by the project.
        """
        
        try:
            return LLM2Vec.from_pretrained(
                self.model_name,
                peft_model_name_or_path=self.supervised_model_name,
                merge_peft=False,
                enable_bidirectional=True,
                pooling_mode=self.pooling_mode,
                max_length=self.max_length,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the LLM2Vec representation with the local llm2vec "
                "loader. This project pins transformers==4.44.2, so avoid loading "
                "remote model code revisions that require newer transformers "
                "modules such as transformers.modeling_layers."
            ) from exc

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a sequence of texts into a NumPy embedding matrix.

        The method converts each input item into a string, replacing ``None``
        values with empty strings, delegates the encoding step to the loaded
        LLM2Vec model, casts the result to ``float32``, and optionally applies
        row-wise L2 normalization.

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
            Any exception raised by the underlying ``self.model.encode`` call
            may propagate.

        Notes
        -----
        This method is the concrete implementation of the abstract
        ``BaseRepresentation.encode`` interface used by downstream components
        that need fixed-size vector representations of text.
        """

        # Convert all inputs into strings so the encoder receives a stable type.
        prepared_texts = ["" if text is None else str(text) for text in texts]
        if not prepared_texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = self.model.encode(prepared_texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if self.normalize:
            # Avoid division by zero when an embedding vector has zero norm.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings