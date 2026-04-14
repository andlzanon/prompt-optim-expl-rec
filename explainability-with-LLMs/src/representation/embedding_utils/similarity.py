import numpy as np

def to_2d_array(vectors: np.ndarray) -> np.ndarray:
    """
    Convert an input array into a two-dimensional float32 NumPy array.

    The function coerces the input into a NumPy array with dtype
    ``np.float32``. If the resulting array is one-dimensional, it is reshaped
    to a single-row matrix so downstream similarity utilities can assume a
    two-dimensional input shape.

    Parameters
    ----------
    vectors : np.ndarray
        Input vector or matrix-like object to be converted.

    Returns
    -------
    np.ndarray
        Two-dimensional NumPy array with dtype ``np.float32``. Higher-
        dimensional inputs are returned unchanged apart from the dtype cast.

    Raises
    ------
    ValueError
        May be propagated by NumPy if the input cannot be converted or
        reshaped as required.

    Notes
    -----
    This helper standardizes array shape and dtype for the embedding utility
    functions that operate on batches of vectors.
    """

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Apply row-wise L2 normalization to a vector matrix.

    The function first ensures the input is represented as a two-dimensional
    float32 array and then divides each row by its Euclidean norm. Rows with
    zero norm are kept unchanged by replacing zero denominators with ``1.0``
    before division.

    Parameters
    ----------
    vectors : np.ndarray
        Input vector or matrix-like object to normalize.

    Returns
    -------
    np.ndarray
        Two-dimensional NumPy array in which each row has unit L2 norm when
        the original norm was non-zero.

    Raises
    ------
    ValueError
        May be propagated by NumPy if the input cannot be converted to the
        expected array representation.

    Notes
    -----
    This helper is used by the cosine-similarity computation to ensure dot
    products correspond to cosine similarity scores.
    """

    vectors = to_2d_array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for rows that are entirely zero.
    norms[norms == 0.0] = 1.0
    return vectors / norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix between two vector sets.

    The function L2-normalizes both inputs row-wise and then returns the
    matrix product between the normalized representations, yielding cosine
    similarities for every pair of rows in ``a`` and ``b``.

    Parameters
    ----------
    a : np.ndarray
        First vector set. One-dimensional inputs are treated as a single row.
    b : np.ndarray
        Second vector set. One-dimensional inputs are treated as a single row.

    Returns
    -------
    np.ndarray
        Matrix whose entry ``[i, j]`` is the cosine similarity between row
        ``i`` of ``a`` and row ``j`` of ``b`` after normalization.

    Raises
    ------
    ValueError
        May be propagated if the inputs cannot be converted into compatible
        matrix shapes for multiplication.

    Notes
    -----
    This function is a core building block for similarity-based selection
    utilities such as the MMR ranking logic.
    """

    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T