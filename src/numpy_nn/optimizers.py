import numpy as np
from numpy.typing import NDArray


class SGD:
    """SGD stands for Stochastic Gradient Descent: It updates the parameter subtracting it
    the gradient multiplied by the learning rate.

    The update is done in place, so the layers see the new values. Gradients
    are not modified by the optimizer.
    """

    def __init__(self, lr: float) -> None:
        """Store the learning rate.

        Args:
            lr: Learning rate.

        Raises:
            ValueError: If ``lr`` is not finite or not positive.
        """
        if not lr > 0 or not np.isfinite(lr):
            raise ValueError(f"Expected a positive finite learning rate, got {lr}")
        self.lr = lr

    def step(
        self, parameters: list[tuple[NDArray[np.float64], NDArray[np.float64] | None]]
    ) -> None:
        """Apply one gradient descent step to every parameter.

        All pairs are validated before any parameter is changed, in order to leave
        the network untouched if there is a fail.

        Args:
            parameters: (parameter, gradient) pairs.

        Raises:
            RuntimeError: If a gradient is None (``backward`` was not called).
            ValueError: If a gradient and its parameter have different shapes.
        """
        parameters = list(parameters)
        for parameter, grad in parameters:
            if grad is None:
                raise RuntimeError("Gradient must not be None.")
            if parameter.shape != grad.shape:
                raise ValueError(
                    f"Expected same shape for parameter and gradient, got {parameter.shape} and {grad.shape}"
                )

        for parameter, grad in parameters:
            parameter -= self.lr * grad
