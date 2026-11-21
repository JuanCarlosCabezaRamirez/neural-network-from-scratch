import numpy as np
import pytest
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose

from numpy_nn.layers import Dense

RTOL, ATOL = 1e-6, 1e-8


def test_backward_known_values():
    layer = Dense(2, 3, rng=np.random.default_rng(0))
    layer.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = np.array([[1.0, 2.0]])
    dy = np.array([[1.0, 0.0, -1.0]])

    layer.forward(x)
    dx = layer.backward(dy)

    assert_allclose(dx, [[-2.0, -2.0]])
    assert_allclose(layer.grad_weights, [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    assert_allclose(layer.grad_biases, [1.0, 0.0, -1.0])


def test_gradient_shapes_match_their_variables():
    rng = np.random.default_rng(1)
    layer = Dense(4, 3, rng=rng)
    x = rng.normal(size=(5, 4))
    layer.forward(x)
    dx = layer.backward(rng.normal(size=(5, 3)))
    assert dx.shape == x.shape
    assert layer.grad_weights.shape == layer.weights.shape
    assert layer.grad_biases.shape == layer.biases.shape


def test_gradient_check():
    rng = np.random.default_rng(2)
    layer = Dense(4, 3, rng=rng)
    layer.biases = rng.normal(size=3)
    x = rng.normal(size=(5, 4))
    g = rng.normal(size=(5, 3))  # plays the role of the upstream gradient

    def loss() -> float:
        return float(np.sum(g * layer.forward(x)))

    # Analytic first: loss() overwrites the cache on every call.
    loss()
    dx = layer.backward(g)
    grad_w, grad_b = layer.grad_weights.copy(), layer.grad_biases.copy()

    assert_allclose(dx, numerical_gradient(loss, x), rtol=RTOL, atol=ATOL)
    assert_allclose(
        grad_w, numerical_gradient(loss, layer.weights), rtol=RTOL, atol=ATOL
    )
    assert_allclose(
        grad_b, numerical_gradient(loss, layer.biases), rtol=RTOL, atol=ATOL
    )


def test_backward_twice_does_not_accumulate():
    rng = np.random.default_rng(3)
    layer = Dense(4, 3, rng=rng)
    layer.forward(rng.normal(size=(5, 4)))
    dy = rng.normal(size=(5, 3))
    layer.backward(dy)
    first = layer.grad_weights.copy()
    layer.backward(dy)
    assert_allclose(layer.grad_weights, first)


def test_backward_does_not_mutate_dy():
    rng = np.random.default_rng(4)
    layer = Dense(4, 3, rng=rng)
    layer.forward(rng.normal(size=(5, 4)))
    dy = rng.normal(size=(5, 3))
    dy_copy = dy.copy()
    layer.backward(dy)
    assert_allclose(dy, dy_copy, rtol=0, atol=0)


def test_backward_before_forward_raises():
    layer = Dense(4, 3, rng=np.random.default_rng(5))
    with pytest.raises(RuntimeError):
        layer.backward(np.zeros((5, 3)))


def test_backward_rejects_wrong_dy_shape():
    rng = np.random.default_rng(6)
    layer = Dense(4, 3, rng=rng)
    layer.forward(rng.normal(size=(5, 4)))
    with pytest.raises(ValueError):
        layer.backward(np.zeros((5, 2)))  # wrong feature dim
    with pytest.raises(ValueError):
        layer.backward(np.zeros((4, 3)))  # wrong batch size
