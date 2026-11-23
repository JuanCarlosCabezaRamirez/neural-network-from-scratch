import numpy as np
from numpy.typing import NDArray


def make_batches(
    n_samples: int, batch_size: int, rng: np.random.Generator
) -> list[NDArray[np.int64]]:
    """Creates a partition of the sample indices, so the mini-batches
    are disjoint and cover all the dataset, which forms an epoch.

    The samples are splitted into batches of the same size, although the
    last batch can be smaller when ``batch_size`` does not divide ``n_samples``.

    The bigger the batches are, the more expensive is running the model steps,
    but with less noisy too. On the other hand, small batches make the steps more
    noisy, but it is less expensive.

    Args:
        n_samples: Number of samples N in the dataset. Must be positive.
        batch_size: Maximum number of indices per batch. Must be positive.
            If it exceeds ``n_samples``, a single batch is returned.
        rng: Generator that draws the permutation. Reusing the same
            generator across epochs gives a different order each epoch.

    Returns:
        List of integer arrays of shape (b_k,), with b_k = batch_size,
        except possibly the last one.

        Raises:
            ValueError: If ``n_samples`` or ``batch_size`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"The number of samples must be positive, got {n_samples}.")
    if batch_size <= 0:
        raise ValueError(f"The batch size must be positive, got {batch_size}.")
    perm = rng.permutation(n_samples)
    return [perm[i : i + batch_size] for i in range(0, n_samples, batch_size)]
