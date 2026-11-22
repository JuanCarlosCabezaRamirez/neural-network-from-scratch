"""Because dense layer forward passes consists of a linear operation, the combination of
forward passes will remain linear. To avoid this, we implement activation functions.

Activation functions are non-linear functions that modify the outputs of a dense layer.
Consequently, we are able to have non-linear outputs in our neural network which enable it
to solve non-linear problems.

In this file, we implement some of the main activation functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReLU:
    """ReLU function computes max{0,a} for each a in x.
    Therefore, every output belongs to [0,+inf).

    ReLU is one of the most popular activation functions. One of the main reasons is its
    simplicity to compute. In addition, its derivative is also straightforward.
    """

    def __init__(self) -> None:
        self._output: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply ReLU element-wise and cache the output.

        Args:
            x: array of any shape, assumed float64.

        Returns:
            New array of the same shape as x, with values in [0, +inf).
        """
        result = np.maximum(x, 0)
        self._output = result
        return result

    def backward(self, dy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backpropagate the loss gradient through ReLU.

        Given dy = dL/dY, the gradient of the loss with respect to the output of this ReLU activation,
        apply the chain rule to obtain the gradients with respect to its input:

        dL/dX = dY * f'(X)

        The product is element-wise because each output depends on a single input.
        For ReLU, its derivative is 1 where x>0 and 0 elsewhere.
        However, ReLU's derivative is undefined at x = 0, since the one-side limits differ when calculate the
        derivative by definition. To avoid this, we define f'(0)=0.
        The mask is computed from the cached output, which is valid because
        relu(x) > 0 exactly when x > 0.

        Args:
            dy: Gradient of the loss with respect to the output, same shape as the output of the last ``forward`` call.

        Returns:
            Gradient of the loss with respect to the input, same shape as the input.

        Raises:
            RuntimeError: If ``forward`` has not been called.
            ValueError: If ``dy`` does not have the required shape.
        """

        if self._output is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        if dy.shape != self._output.shape:
            raise ValueError(
                f"Expected dy of shape {self._output.shape}, got {dy.shape}."
            )

        f_derivative = np.where(self._output > 0, 1, 0)
        dx = dy * f_derivative
        return dx

    def parameters(
        self,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64] | None]]:
        """Return an empty list, as ReLU does not have trainable parameters.

        It exists so that ``Sequential`` can treat every layer in the same way.
        """
        return []


class Sigmoid:
    """Sigmoid function computes 1/(1+exp(-a)) for each a in x.

    The returning value belongs to the interval (0,1).

    In order to avoid overflow problems, we distinguish between two cases:

    If a is non negative we use the aforementioned formula, as Euler's number
    raised to minus large values results in really small numbers, this leads to underflow which
    is definitely not a problem because NumPy ignores it by default, rounding  to 0 (the real value
    is indistinguishable from 0 at that scale). The real problem would be overflow, as it produces inf.

    However, if a is negative we will use the equivalent Sigmoid formula: exp(a)/(1+exp(a)),
    resulting of multiplying sigmoid formula by exp(a)/exp(a).
    In this case, we also avoid overflow problem, as we compute the exponential raised to negative values.

    Sigmoid function can be denote as σ.
    """

    def __init__(self) -> None:
        self._output: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Sigmoid function element-wise and cache the output.

        Args:
            x: array of any shape, assumed float64.

        Returns:
            New array of the same shape as x, with values in (0,1).
        """
        x = np.asarray(x, dtype=np.float64)
        positives = x >= 0
        result = np.empty_like(x)
        result[positives] = 1 / (1 + np.exp(-x[positives]))
        neg_values = x[~positives]
        exp_neg = np.exp(neg_values)
        result[~positives] = exp_neg / (1 + exp_neg)

        self._output = result
        return result

    def backward(self, dy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backpropagate the loss gradient through Sigmoid.

        Given dy = dL/dY, the gradient of the loss with respect to the output of this Sigmoid activation,
        apply the chain rule to obtain the gradients with respect to its input:

        dL/dX = dY * f'(X)

        The product is element-wise because each output depends on a single input.
        For Sigmoid, its derivative is: σ'(x) = σ(x)(1-σ(x)).

        Args:
             dy: Gradient of the loss with respect to the output, same shape as the output of the last ``forward`` call.

        Returns:
            Gradient of the loss with respect to the input, same shape as the input.

        Raises:
            RuntimeError: If ``forward`` has not been called.
            ValueError: If ``dy`` does not have the required shape.
        """

        if self._output is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        if dy.shape != self._output.shape:
            raise ValueError(
                f"Expected dy of shape {self._output.shape}, got {dy.shape}."
            )

        f_derivative = self._output * (1 - self._output)
        dx = dy * f_derivative
        return dx

    def parameters(
        self,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64] | None]]:
        """Return an empty list, as Sigmoid does not have trainable parameters.

        It exists so that ``Sequential`` can treat every layer in the same way.
        """
        return []


class Tanh:
    """The hyperbolic tangent function computes sinh(a)/cosh(a)=(exp(a)-exp(-a))/(exp(a)+exp(-a)).
    As a result, its image is (-1,1)

    In this function, we do not have to distinguish cases because tanh is well implemented in NumPy.
    """

    def __init__(self) -> None:
        self._output: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the hyperbolic tangent element-wise and cache the output.

        Args:
            x: array of any shape, assumed float64.

        Returns:
            New array of the same shape as x, with values in (-1,1).
        """

        result = np.tanh(x)
        self._output = result
        return result

    def backward(self, dy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backpropagate the loss gradient through Tanh.

        Given dy = dL/dY, the gradient of the loss with respect to the output of this hyperbolic tangent
        activation, apply the chain rule to obtain the gradients with respect to its input:

        dL/dX = dY * f'(X)

        The product is element-wise because each output depends on a single input.
        For hyperbolic tangent, its derivative is: tanh'(x) = 1 - tanh^2(x).

        Args:
            dy: Gradient of the loss with respect to the output, same shape as the output of the last ``forward`` call.

        Returns:
            Gradient of the loss with respect to the input, same shape as the input.

        Raises:
            RuntimeError: If ``forward`` has not been called.
            ValueError: If ``dy`` does not have the required shape.
        """

        if self._output is None:
            raise RuntimeError("Backward pass must be done after forward pass.")

        if dy.shape != self._output.shape:
            raise ValueError(
                f"Expected dy of shape {self._output.shape}, got {dy.shape}."
            )

        f_derivative = 1 - self._output**2
        dx = dy * f_derivative
        return dx

    def parameters(
        self,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64] | None]]:
        """Return an empty list, as hypeborlic tangent does not have trainable parameters.

        It exists so that ``Sequential`` can treat every layer in the same way.
        """
        return []
