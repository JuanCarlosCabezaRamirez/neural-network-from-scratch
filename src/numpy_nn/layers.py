from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Dense:
    """Fully connected layer that computes ``Y = x @ W + b``.

    Every neuron of n_neurons computes a weighted sum of the inputs, plus the bias.

    Weights are drawn from a Gaussian distribution with mean 0 and standard deviation of 1/sqrt(n_inputs)
    to avoid neuron's numbers growing without bound with the layer width.
    Biases start at zero.
    """

    def __init__(
        self, n_inputs: int, n_neurons: int, rng: np.random.Generator | None = None
    ) -> None:
        """Initialize the parameters.

        Args:
            n_inputs: Number of input features.
            n_neurons: Number of neurons (output features).
            rng: Random generator for reproducibility. If None, rng = np.random.default_rng() is used.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.weights = rng.normal(0, 1 / np.sqrt(n_inputs), size=(n_inputs, n_neurons))
        self.biases = np.zeros((n_neurons,))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self._x: NDArray[np.float64] | None = None
        self.grad_weights: NDArray[np.float64] | None = None
        self.grad_biases: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the layer output for a batch of samples and stores a reference to x,
        so modifying x in place before backward gives wrong gradients.

        Args:
            x: Input batch, shape (N, n_inputs), one sample per row.

        Returns:
            Array of shape (N, n_neurons).

        Raises:
            ValueError: If ``x`` is not 2D or its second dimension is not n_inputs.
        """
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got {x.ndim}D array with shape {x.shape}."
            )

        elif x.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected ({x.shape[0]}, {self.n_inputs}) shape, got ({x.shape[0]}, {x.shape[1]})."
            )

        self._x = x

        return x @ self.weights + self.biases

    def backward(self, dy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backpropagate the loss gradient through the layer.

        Given dy = dL/dY, the gradient of the loss with respect to this layer's
        output, apply the chain rule to obtain the gradients with respect to the
        parameters and the input. For Y = X @ W + b:

            dL/dW = X.T @ dY          (n_inputs, n_neurons)
            dL/db = dY summed over N  (n_neurons,)
            dL/dX = dY @ W.T          (N, n_inputs)

        The parameter gradients are stored in ``grad_weights`` and
        ``grad_biases``, overwriting any previous values so they do not
        accumulate.

        Args:
            dy: Gradient of the loss with respect to the layer output, shape (N, n_neurons).

        Returns:
            Gradient of the loss with respect to the layer input, shape (N, n_inputs).

        Raises:
            RuntimeError: If ``forward`` has not been called.
            ValueError: If ``dy`` does not have the required shape.
        """
        if self._x is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        expected_shape = (self._x.shape[0], self.n_neurons)
        if dy.shape != expected_shape:
            raise ValueError(f"Expected dy of shape {expected_shape}, got {dy.shape}.")

        self.grad_weights = self._x.T @ dy
        self.grad_biases = dy.sum(axis=0)
        dx = dy @ self.weights.T
        return dx
