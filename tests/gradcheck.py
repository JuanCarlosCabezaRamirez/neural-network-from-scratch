"""Finite-difference gradient estimation, used only by the tests."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def numerical_gradient(
    f: Callable[[], float], array: NDArray[np.float64], eps: float = 1e-6
) -> NDArray[np.float64]:
    """Estimate the gradient of the scalar function f with respect to `array`.

    Uses central differences. `array` is perturbed IN PLACE, one entry at a
    time, and restored afterwards, so `f` (a closure that reads the array,
    directly or through a layer) sees the perturbed values.

    Args:
        f: Zero-argument callable returning a float.
        array: Float64 array of any shape; the variable to differentiate.
        eps: Perturbation size.

    Returns:
        Array of the same shape as `array` with the estimated gradient.
    """
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        f_plus = f()
        array[idx] = original - eps
        f_minus = f()
        array[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad
