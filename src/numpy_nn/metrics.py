import numpy as np
from numpy.typing import NDArray


def accuracy(logits: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Fraction of samples whose highest-scoring class equals the label.

    The prediction is the argmax of each row of logits. Softmax is not needed,
    as it does not change which entry is largest. If there is a tie, the prediction
    will be the first maximal class because it is NumPy's argmax behavior.

    Args:
        logits: Raw scores of shape (N, C).
        labels: Integer class indices of shape (N,).

    Returns:
        Accuracy in [0, 1] as a Python float.

    Raises:
        ValueError: If logits are not a non-empty 2D array, if labels do
            not have shape (N,), or if labels are not of integer dtype.
    """
    if logits.size == 0 or logits.ndim != 2:
        raise ValueError(
            f"Logits must be a non-empty 2D array, got shape {logits.shape}."
        )
    if labels.shape != (logits.shape[0],):
        raise ValueError(
            f"Labels must be an array of shape ({logits.shape[0]},), got shape {labels.shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"Labels can only contain integers, got that labels has type: {labels.dtype}."
        )
    pred = np.argmax(logits, axis=1)

    return float((pred == labels).mean())
