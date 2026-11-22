"""Tests for the Sequential container."""

import numpy as np
import pytest
from gradcheck import (
    numerical_gradient,
)  # import it the same way as in test_loss_chain.py

from numpy_nn.activations import ReLU, Sigmoid, Tanh
from numpy_nn.layers import Dense
from numpy_nn.losses import MSELoss, SoftmaxCrossEntropy
from numpy_nn.sequential import Sequential


def make_net(activation_cls, seed=0):
    rng = np.random.default_rng(seed)
    return Sequential([Dense(3, 5, rng=rng), activation_cls(), Dense(5, 4, rng=rng)])


# --- construction -----------------------------------------------------------


def test_empty_layer_list_raises():
    with pytest.raises(ValueError):
        Sequential([])


# --- forward / backward plumbing ---------------------------------------------


def test_forward_equals_manual_chain():
    rng = np.random.default_rng(1)
    d1, act, d2 = Dense(3, 5, rng=rng), Tanh(), Dense(5, 2, rng=rng)
    net = Sequential([d1, act, d2])
    x = rng.normal(size=(6, 3))

    expected = d2.forward(act.forward(d1.forward(x)))
    np.testing.assert_allclose(net.forward(x), expected)


def test_backward_returns_gradient_with_input_shape():
    net = make_net(Tanh)
    x = np.random.default_rng(2).normal(size=(6, 3))
    y = net.forward(x)
    dx = net.backward(np.ones_like(y))
    assert dx.shape == x.shape


def test_backward_before_forward_raises():
    net = make_net(Tanh)
    with pytest.raises(RuntimeError):
        net.backward(np.ones((6, 4)))


# --- parameters --------------------------------------------------------------


def test_activations_have_no_parameters():
    for act in (ReLU(), Sigmoid(), Tanh()):
        assert act.parameters() == []


def test_parameters_are_the_same_objects_as_the_layers():
    rng = np.random.default_rng(3)
    d1, d2 = Dense(3, 5, rng=rng), Dense(5, 4, rng=rng)
    net = Sequential([d1, Tanh(), d2])

    params = net.parameters()
    assert len(params) == 4  # weights and biases of each Dense, in layer order
    assert params[0][0] is d1.weights
    assert params[1][0] is d1.biases
    assert params[2][0] is d2.weights
    assert params[3][0] is d2.biases


def test_gradients_are_none_before_backward():
    net = make_net(Tanh)
    assert all(grad is None for _, grad in net.parameters())


def test_gradients_are_the_layers_gradient_objects_after_backward():
    rng = np.random.default_rng(4)
    d1, d2 = Dense(3, 5, rng=rng), Dense(5, 4, rng=rng)
    net = Sequential([d1, Tanh(), d2])

    y = net.forward(rng.normal(size=(6, 3)))
    net.backward(np.ones_like(y))

    params = (
        net.parameters()
    )  # re-read AFTER backward: grad objects are replaced each pass
    assert params[0][1] is d1.grad_weights
    assert params[1][1] is d1.grad_biases
    assert params[2][1] is d2.grad_weights
    assert params[3][1] is d2.grad_biases


# --- gradients of the whole container against finite differences ---------------


@pytest.mark.parametrize("activation_cls", [Tanh, Sigmoid, ReLU])
@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_container_gradients_match_numerical(activation_cls, loss_name):
    rng = np.random.default_rng(5)
    net = make_net(activation_cls, seed=6)
    x = rng.normal(size=(7, 3))
    x = np.where(np.abs(x) < 0.2, 0.2, x)  # stay away from ReLU's kink at the input

    if loss_name == "mse":
        loss = MSELoss()
        target = rng.normal(size=(7, 4))
    else:
        loss = SoftmaxCrossEntropy()
        target = rng.integers(0, 4, size=7)

    def loss_value():
        return loss.forward(net.forward(x), target)

    # analytic backward FIRST: every call to loss_value overwrites the caches
    loss_value()
    dx = net.backward(loss.backward())
    analytic = [grad.copy() for _, grad in net.parameters()]
    analytic_dx = dx.copy()

    for (param, _), grad in zip(net.parameters(), analytic):
        assert np.any(grad != 0)  # not trivially zero
        numeric = numerical_gradient(loss_value, param)
        np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)

    numeric_dx = numerical_gradient(loss_value, x)
    np.testing.assert_allclose(analytic_dx, numeric_dx, rtol=1e-6, atol=1e-8)
