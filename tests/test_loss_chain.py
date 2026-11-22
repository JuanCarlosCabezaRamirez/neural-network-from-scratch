"""End-to-end gradient check: Dense -> activation -> Dense -> loss."""

import numpy as np
import pytest
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose

from numpy_nn.activations import ReLU, Sigmoid, Tanh
from numpy_nn.layers import Dense
from numpy_nn.losses import MSELoss, SoftmaxCrossEntropy

RTOL, ATOL = 1e-6, 1e-8


@pytest.mark.parametrize("activation_cls", [Tanh, Sigmoid, ReLU])
@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_full_chain_gradients(activation_cls, loss_name):
    rng = np.random.default_rng(0)
    n_samples, n_in, n_hidden, n_out = 6, 3, 5, 4
    layers = [Dense(n_in, n_hidden, rng), activation_cls(), Dense(n_hidden, n_out, rng)]
    x = rng.normal(size=(n_samples, n_in))
    if loss_name == "mse":
        loss_fn, target = MSELoss(), rng.normal(size=(n_samples, n_out))
    else:
        loss_fn, target = SoftmaxCrossEntropy(), rng.integers(0, n_out, size=n_samples)

    def loss():
        out = x
        for layer in layers:
            out = layer.forward(out)
        return loss_fn.forward(out, target)

    # Analytic backward FIRST: every call to loss() overwrites the caches.
    loss()
    grad = loss_fn.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)
    dx = grad

    checks = {
        "dW1": (layers[0].weights, layers[0].grad_weights),
        "db1": (layers[0].biases, layers[0].grad_biases),
        "dW2": (layers[2].weights, layers[2].grad_weights),
        "db2": (layers[2].biases, layers[2].grad_biases),
        "dx": (x, dx),
    }
    for name, (array, analytic) in checks.items():
        assert analytic.shape == array.shape, name
        assert np.linalg.norm(analytic) > 1e-6, f"{name} is trivially zero"
        numeric = numerical_gradient(loss, array)
        assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL, err_msg=name)
