import numpy as np
from numpy.typing import NDArray


class Sequential:
    """Container that chains layers into a network.

    The container expects the layer types to have ``forward``, ``backward``
    and ``parameters``.

    It keeps its own list of layers, but the layer objects are shared with the
    caller.
    """

    def __init__(self, layers: list) -> None:
        """Store the layers.

        Args:
            layers: List of layers (Dense or activations), in forward order.

        Raises:
            ValueError: If ``layers`` is empty.
        """
        if not layers:
            raise ValueError("Layers must be a non-empty list.")

        self._layers = list(layers)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run the input through every layer, starting from the first layer up to the last one.

        Args:
            x: Input batch, shape (N, n_inputs of the first layer).

        Returns:
            Output of the last layer, shape (N, n_neurons of the last Dense).

        Raises:
            ValueError: Raised from the layers if a shape is wrong.
        """
        current_x = x
        for layer in self._layers:
            current_x = layer.forward(current_x)
        return current_x

    def backward(self, dy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backpropagate through every layer, starting from the last layer up to the first one.

        Args:
            dy: Gradient of the loss with respect to the network output,
                same shape as the output of the last ``forward`` call.

        Returns:
            Gradient of the loss with respect to the network input,
            same shape as the input of the last ``forward`` call.

        Raises:
            RuntimeError: Raised from the layers if ``forward`` has not been called.
            ValueError: Raised from the layers if ``dy`` has the wrong shape.
        """
        current_dx = dy
        for layer in reversed(self._layers):
            current_dx = layer.backward(current_dx)
        return current_dx

    def parameters(
        self,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64] | None]]:
        """List every trainable parameter together with its gradient.

        The list is built at each call in layer order (weights, then biases,
        for each Dense). The arrays have the layers' own objects, not copies,
        so modifying them changes the network. Thus, this implies that each
        ``backward`` replaces the gradient arrays, which are None if ``backward``
        has not been called yet.

        Returns:
            Flat list of (parameter, gradient) pairs.
        """
        params = []
        for layer in self._layers:
            params += layer.parameters()
        return params
