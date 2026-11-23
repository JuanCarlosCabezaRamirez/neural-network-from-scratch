import numpy as np
from numpy.typing import NDArray

from numpy_nn.data import make_batches
from numpy_nn.losses import MSELoss, SoftmaxCrossEntropy
from numpy_nn.optimizers import SGD
from numpy_nn.sequential import Sequential


def train(
    model: Sequential,
    loss: MSELoss | SoftmaxCrossEntropy,
    optimizer: SGD,
    x: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64],
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[float]:
    """Train ``model`` with mini-batch gradient descent.

    Each epoch draws a shuffle of the sample indices. For every batch
    it runs forward, loss, backward and one optimizer step, in that order.
    The dataset is never modified.

    The reported epoch loss is the sample-weighted mean (1/N) * sum_k
    b_k * L_k, so a smaller last batch would count proportionally.

    Args:
        model: Network to train. Its parameters are updated in place.
        loss: Loss whose ``forward`` returns the batch mean.
        optimizer: Applies one update per batch.
        x: Inputs of shape (N, n_features).
        y: Targets with N rows: float targets for MSELoss, integer class
            indices of shape (N,) for SoftmaxCrossEntropy.
        epochs: Number of passes over the data. Must be a positive int.
        batch_size: Samples per batch. Must be positive (checked by
            ``make_batches``).
        rng: Generator used for shuffling.

    Returns:
        Mean training loss per epoch, a list of ``epochs`` floats.

    Raises:
        TypeError: If ``epochs`` is not an int.
        ValueError: If ``epochs`` is not positive, ``x`` is not 2D, ``y``
            and ``x`` have different numbers of rows, or ``batch_size``
            or the dataset size is not positive.
    """
    if not isinstance(epochs, int):
        raise TypeError(f"Epochs must be an integer, got {type(epochs)}.")
    if epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {epochs}.")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D array, got {x.ndim}D array.")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y must have the same rows as x, got {y.shape[0]} rows from y and {x.shape[0]} from x."
        )
    history: list[float] = []
    for _ in range(epochs):
        loss_sum = 0.0
        batches = make_batches(x.shape[0], batch_size, rng)
        for batch in batches:
            loss_sum += loss.forward(model.forward(x[batch]), y[batch]) * batch.size
            model.backward(loss.backward())
            optimizer.step(model.parameters())
        history.append(loss_sum / x.shape[0])

    return history
