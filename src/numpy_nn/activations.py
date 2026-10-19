
"""Because dense layer forward passes consists of a linear operation, the combination of 
forward passes will remain linear. To avoid this, we implement activation functions.

Activation functions are non-linear functions that modify the outputs of a dense layer.
Consequently, we are able to have non-linear outputs in our neural network what enables it
to resolve non-linear problems.

In this file, we implement some of the main activation functions.
"""

import numpy as np
from numpy.typing import NDArray


class ReLU:
    """ReLU function computes max{0,a} for each a in x.
    Therefore, every output belongs to [0,+inf).

    ReLU is one of the most popular activation functions. One of the main reasons is its 
    simplicity to compute. In addition, its derivative is also straightforward.
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply ReLU element-wise.

        Args:
            x: array of any shape, assumed float64.

        Returns:
            New array of the same shape as x, with values in [0, +inf).
        """
        return np.maximum(x, 0)


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
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Sigmoid function element-wise.

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

        return result


class Tanh:
    """The hyperbolic tangent function computes sinh(a)/cosh(a)=(exp(a)-exp(-a))/(exp(a)+exp(-a)).
    As a result, its image is (-1,1)

    In this function, we do not have to distinguish cases because tanh is well implemented in NumPy.
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the hyperbolic tangent element-wise.

        Args:
            x: array of any shape, assumed float64.

        Returns:
            New array of the same shape as x, with values in (-1,1).
        """
        return np.tanh(x)
