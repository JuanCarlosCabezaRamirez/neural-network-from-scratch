import numpy as np
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose

from numpy_nn.activations import Tanh
from numpy_nn.layers import Dense


def test_dense_tanh_dense_gradients():
    rng = np.random.default_rng(0)
    d1, act, d2 = Dense(4, 5, rng=rng), Tanh(), Dense(5, 2, rng=rng)
    x = rng.normal(size=(6, 4))
    g = rng.normal(size=(6, 2))

    def loss() -> float:
        return float(np.sum(g * d2.forward(act.forward(d1.forward(x)))))

    loss()
    dx = d1.backward(act.backward(d2.backward(g)))
    analytic = [
        (d1.weights, d1.grad_weights.copy()),
        (d1.biases, d1.grad_biases.copy()),
        (d2.weights, d2.grad_weights.copy()),
        (d2.biases, d2.grad_biases.copy()),
        (x, dx),
    ]
    for array, grad in analytic:
        assert_allclose(grad, numerical_gradient(loss, array), rtol=1e-6, atol=1e-8)
