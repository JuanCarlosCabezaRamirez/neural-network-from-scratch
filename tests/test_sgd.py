"""Tests for the SGD optimizer."""

import numpy as np
import pytest

from numpy_nn.activations import Tanh
from numpy_nn.layers import Dense
from numpy_nn.losses import MSELoss, SoftmaxCrossEntropy
from numpy_nn.optimizers import SGD
from numpy_nn.sequential import Sequential

# --- construction -----------------------------------------------------------


@pytest.mark.parametrize("lr", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_invalid_learning_rate_raises(lr):
    with pytest.raises(ValueError):
        SGD(lr)


def test_valid_learning_rate_is_stored():
    assert SGD(0.5).lr == 0.5


# --- the update rule ----------------------------------------------------------


def test_known_values():
    w = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    SGD(0.1).step([(w, g)])
    # [1 - 0.1*0.5, 2 - 0.1*(-1)] = [0.95, 2.1]
    np.testing.assert_allclose(w, [0.95, 2.1])


def test_all_pairs_are_updated():
    w1, g1 = np.ones((2, 3)), np.full((2, 3), 2.0)
    w2, g2 = np.zeros(3), np.array([1.0, 2.0, 3.0])
    SGD(0.5).step([(w1, g1), (w2, g2)])
    np.testing.assert_allclose(w1, np.zeros((2, 3)))
    np.testing.assert_allclose(w2, [-0.5, -1.0, -1.5])


def test_update_is_in_place():
    w = np.array([1.0, 2.0])
    alias = w  # a second name for the same array
    SGD(0.1).step([(w, np.array([1.0, 1.0]))])
    assert alias is w
    np.testing.assert_allclose(alias, [0.9, 1.9])


def test_gradients_are_not_modified():
    w = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    g_before = g.copy()
    SGD(0.1).step([(w, g)])
    np.testing.assert_array_equal(g, g_before)


def test_zero_gradient_leaves_parameters_unchanged():
    w = np.array([1.0, 2.0])
    SGD(0.1).step([(w, np.zeros(2))])
    np.testing.assert_array_equal(w, [1.0, 2.0])


# --- error conditions ----------------------------------------------------------


def test_none_gradient_raises_runtime_error():
    with pytest.raises(RuntimeError):
        SGD(0.1).step([(np.ones(2), None)])


def test_step_before_backward_on_a_network_raises():
    net = Sequential([Dense(3, 2, rng=np.random.default_rng(0))])
    with pytest.raises(RuntimeError):
        SGD(0.1).step(net.parameters())


def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        SGD(0.1).step([(np.ones((2, 3)), np.ones((3, 2)))])


def test_failed_step_changes_nothing():
    w1 = np.array([1.0, 2.0])
    w2 = np.array([3.0, 4.0])
    with pytest.raises(RuntimeError):
        SGD(0.1).step([(w1, np.ones(2)), (w2, None)])
    np.testing.assert_array_equal(w1, [1.0, 2.0])  # the valid pair was not applied

    with pytest.raises(ValueError):
        SGD(0.1).step([(w1, np.ones(2)), (w2, np.ones(3))])
    np.testing.assert_array_equal(w1, [1.0, 2.0])


# --- the step on a real network ------------------------------------------------


def make_problem(loss_name):
    rng = np.random.default_rng(0)
    net = Sequential([Dense(3, 5, rng=rng), Tanh(), Dense(5, 4, rng=rng)])
    x = rng.normal(size=(8, 3))
    if loss_name == "mse":
        return net, MSELoss(), x, rng.normal(size=(8, 4))
    return net, SoftmaxCrossEntropy(), x, rng.integers(0, 4, size=8)


def loss_and_gradients(net, loss, x, target):
    value = loss.forward(net.forward(x), target)
    net.backward(loss.backward())
    return value


@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_small_step_decreases_loss(loss_name):
    net, loss, x, target = make_problem(loss_name)
    before = loss_and_gradients(net, loss, x, target)
    grad_norm_sq = sum(np.sum(g**2) for _, g in net.parameters())
    assert grad_norm_sq > 0  # otherwise the test proves nothing

    SGD(1e-3).step(net.parameters())

    after = loss.forward(net.forward(x), target)
    assert after < before


@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_tiny_step_matches_first_order_prediction(loss_name):
    # L(w - lr*g) - L(w) ≈ -lr * ||g||^2, with relative error O(lr)
    net, loss, x, target = make_problem(loss_name)
    lr = 1e-5
    before = loss_and_gradients(net, loss, x, target)
    grad_norm_sq = sum(np.sum(g**2) for _, g in net.parameters())

    SGD(lr).step(net.parameters())

    after = loss.forward(net.forward(x), target)
    np.testing.assert_allclose(after - before, -lr * grad_norm_sq, rtol=1e-2)


def test_network_sees_the_updated_weights():
    net, loss, x, target = make_problem("mse")
    loss_and_gradients(net, loss, x, target)
    weights_before = net.parameters()[0][0].copy()

    SGD(0.1).step(net.parameters())

    weights_after = net.parameters()[0][0]  # the layer's own array, re-read
    assert not np.array_equal(weights_before, weights_after)


def test_step_accepts_a_one_shot_iterator():
    w = np.array([1.0, 2.0])
    SGD(0.1).step((p, g) for p, g in [(w, np.array([1.0, 1.0]))])
    np.testing.assert_allclose(w, [0.9, 1.9])
