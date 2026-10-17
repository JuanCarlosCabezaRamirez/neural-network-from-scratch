from __future__ import annotations

import numpy as np


class Dense:
    """Fully connected layer that computes ``z = x @ W + b``.

    Every neuron of n_neurons computes a weighted sum of the inputs, plus the bias.

    Weights are drawn from a Gaussian distribution with mean 0 and desviation of 1/sqrt(n_inputs)
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a batch of samples.

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

        return x @ self.weights + self.biases
