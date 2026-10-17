"""Tests for numpy_nn.layerspy"""

import numpy as np
import pytest

from numpy_nn.layers import Dense


def make_layer(in_f: int = 2, out_f: int = 3, seed: int = 0) -> Dense:
    return Dense(in_f, out_f, rng=np.random.default_rng(seed))


# ---------- initialization ----------


def test_parameter_shapes():
    layer = make_layer(4, 5)
    assert layer.weights.shape == (4, 5)
    assert layer.biases.shape == (5,)


def test_bias_starts_at_zero():
    layer = make_layer(4, 5)
    assert np.all(layer.biases == 0.0)


def test_init_is_reproducible_with_same_seed():
    a = make_layer(6, 7, seed=42)
    b = make_layer(6, 7, seed=42)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_init_differs_across_seeds():
    a = make_layer(6, 7, seed=1)
    b = make_layer(6, 7, seed=2)
    assert not np.allclose(a.weights, b.weights)


def test_init_scale_is_about_one_over_sqrt_fan_in():
    # With many weights, the sample std should be close to 1/sqrt(in_features).
    in_f = 400
    layer = Dense(in_f, 400, rng=np.random.default_rng(0))
    assert layer.weights.std() == pytest.approx(1 / np.sqrt(in_f), rel=0.05)


def test_default_rng_is_accepted():
    layer = Dense(3, 2)  # no rng passed
    assert layer.weights.shape == (3, 2)


# ---------- forward pass ----------


def test_output_shape():
    layer = make_layer(4, 5)
    x = np.random.default_rng(1).normal(size=(10, 4))
    assert layer.forward(x).shape == (10, 5)


def test_known_values():
    # Hand-computed example: 2 inputs -> 3 neurons, batch of 3.
    layer = make_layer(2, 3)
    layer.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.biases = np.array([0.5, -1.0, 0.0])
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([[1.5, 1.0, 3.0], [4.5, 4.0, 6.0], [5.5, 6.0, 9.0]])
    np.testing.assert_allclose(layer.forward(x), expected)


def test_zero_input_returns_bias_for_every_row():
    layer = make_layer(3, 4)
    layer.biases = np.array([1.0, -2.0, 3.0, 0.5])
    out = layer.forward(np.zeros((5, 3)))
    np.testing.assert_allclose(out, np.tile(layer.biases, (5, 1)))


def test_batch_matches_single_samples():
    layer = make_layer(4, 3)
    x = np.random.default_rng(3).normal(size=(6, 4))
    batch_out = layer.forward(x)
    for i in range(x.shape[0]):
        single = layer.forward(x[i : i + 1])
        np.testing.assert_allclose(batch_out[i], single[0])


def test_affine_property():

    layer = make_layer(4, 3)
    layer.biases = np.array([0.3, -0.7, 1.1])
    rng = np.random.default_rng(4)
    x1 = rng.normal(size=(5, 4))
    x2 = rng.normal(size=(5, 4))
    lhs = layer.forward(x1 + x2)
    rhs = layer.forward(x1) + layer.forward(x2) - layer.biases
    np.testing.assert_allclose(lhs, rhs)


def test_forward_does_not_modify_input_or_parameters():
    layer = make_layer(4, 3)
    x = np.random.default_rng(5).normal(size=(2, 4))
    x_copy, W_copy, b_copy = x.copy(), layer.weights.copy(), layer.biases.copy()
    layer.forward(x)
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(layer.weights, W_copy)
    np.testing.assert_array_equal(layer.biases, b_copy)


# ---------- input validation ----------


def test_rejects_1d_input():
    layer = make_layer(4, 3)
    with pytest.raises(ValueError):
        layer.forward(np.zeros(4))


def test_rejects_wrong_feature_dimension():
    layer = make_layer(4, 3)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((2, 5)))
