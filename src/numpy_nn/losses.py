"""Loss functions.

Each loss maps the pair (predictions, targets) to a scalar with ``forward`` and returns
dL/d(predictions) with ``backward``.

``backward`` takes no argument because backpropagation starts here (dL/dL = 1).
"""

import numpy as np
from numpy.typing import NDArray


class MSELoss:
    """Mean squared error, averaged over all elements.

    With M = y_pred.size:
        L = (1/M) * sum((y_pred - y_true) ** 2)
        dL/dy_pred = 2 * (y_pred - y_true) / M
    Losses average over the batch: the 1/M factor is here, not in the layers.

    Cache: ``_diff`` = y_pred - y_true, a new array computed in ``forward``.
    It is None until ``forward`` is called.
    """

    def __init__(self) -> None:
        self._diff: NDArray[np.float64] | None = None

    def forward(
        self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]
    ) -> float:
        """Compute the mean squared error.

        Args:
            y_pred: Predictions, any shape. Converted to a numpy array of float64.
            y_true: Targets, same shape as ``y_pred``. Converted to a numpy array of float64.

        Returns:
            The loss as a float.

        Raises:
            ValueError: If the arrays are empty or their shapes differ.
        """

        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        if y_pred.size == 0 or y_true.size == 0:
            raise ValueError("Arrays cannot be empty.")

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Expected the same shapes, got {y_true.shape} and {y_pred.shape}"
            )

        self._diff = y_pred - y_true
        return float(np.mean(self._diff**2))

    def backward(self) -> NDArray[np.float64]:
        """Compute the gradient of the loss with respect to the predictions.

        Returns:
            dL/dy_pred, a new array with the same shape as ``y_pred``.

        Raises:
            RuntimeError: If ``forward`` has not been called.
        """
        if self._diff is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        return 2 * self._diff / self._diff.size


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy.

    For logits Z of shape (N, C) and integer labels y of shape (N,):

        log p = (z - max(z)) - log(sum(exp(z - max(z))))   (per row)
        L = -(1/N) * sum_n log p[n, y_n]
        dL/dZ = (P - onehot(y)) / N

    Subtracting max(z) in each row does not change the probabilities, but avoid that the
    log-sum-exp form exponentiates a positive number, making it stable for extreme logits.

    Cache: ``_probs`` (N, C) holds the softmax probabilities and ``_labels`` (N,) the labels,
    stored by reference. Do not modify the labels array between ``forward`` and ``backward``.
    Both are None until ``forward`` is called.
    """

    def __init__(self) -> None:
        self._probs: NDArray[np.float64] | None = None
        self._labels: NDArray[np.integer] | None = None

    def forward(
        self, logits: NDArray[np.float64], labels: NDArray[np.integer]
    ) -> float:
        """Compute the mean cross-entropy of softmax(logits) against labels.

        Args:
            logits: Raw scores, shape (N, C), converted to a numpy array of float64, N >= 1 and C >= 1.
            labels: Class indices, shape (N,), converted to a numpy array which must contain only integer dtype, each in [0, C - 1].

        Returns:
            The loss as a float.

        Raises:
            ValueError: If logits are not a non-empty 2D array, labels do not have shape (N,),
            labels are not integers or any label is outside [0, C - 1].
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)  # We will check that all labels dtype are integers

        if logits.size == 0 or logits.ndim != 2:
            raise ValueError("Logits must be a non-empty 2D array.")

        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError(
                "Labels must be a 1D array with the same number of entries as logits' number of rows."
            )

        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must exclusively contain integers.")

        if not np.all((0 <= labels) & (labels <= logits.shape[1] - 1)):
            raise ValueError(
                "All labels' values must be between 0 and the number of logits' columns minus one."
            )

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        values = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

        log_probs = shifted - values
        probs = np.exp(log_probs)
        self._probs = probs
        self._labels = labels

        return float(-log_probs[np.arange(log_probs.shape[0]), labels].mean())

    def backward(self) -> NDArray[np.float64]:
        """Compute the gradient of the loss with respect to the logits.

        Returns:
            dL/dZ = (P - onehot(y)) / N, a new array with shape (N, C).
            The cached probabilities are not modified.

        Raises:
            RuntimeError: If ``forward`` has not been called.
        """
        if self._probs is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        probs = self._probs.copy()
        probs[np.arange(probs.shape[0]), self._labels] = (
            probs[np.arange(probs.shape[0]), self._labels] - 1
        )

        return probs / self._labels.shape[0]
