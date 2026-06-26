from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class BaseRepresentation(ABC):
    """
    Abstract interface for text-representation backends.

    This base class defines the minimal contract expected from representation
    components that transform one or more text inputs into numeric vectors.
    Concrete subclasses are responsible for providing the actual encoding
    strategy.

    Notes
    -----
    Because the class contains an abstract method, it is intended to be
    subclassed rather than instantiated directly.
    """

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Convert a sequence of text inputs into a NumPy array representation.

        This method defines the core interface that all concrete
        representation classes must implement. It serves as the common entry
        point used by the rest of the codebase to obtain vector
        representations from textual content.

        Parameters
        ----------
        texts : Sequence[str]
            Sequence of text strings to be encoded.

        Returns
        -------
        np.ndarray
            NumPy array containing the encoded representation produced by the
            concrete implementation.
        """

        pass