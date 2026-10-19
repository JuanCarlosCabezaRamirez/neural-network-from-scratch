import numpy as np
import pytest

from numpy_nn.activations import ReLU, Sigmoid, Tanh

ACTIVATIONS = [ReLU, Sigmoid, Tanh]


@pytest.fixture
def x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(5, 4))


# ---------- properties shared by every activation ----------


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_preserves_shape(act_cls):
    for shape in [(3,), (5, 4), (2, 3, 4)]:
        arr = np.zeros(shape)
        assert act_cls().forward(arr).shape == shape


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_does_not_mutate_input(act_cls, x):
    original = x.copy()
    act_cls().forward(x)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_is_elementwise(act_cls, x):
    # f applied to the whole array equals f applied entry by entry.
    act = act_cls()
    full = act.forward(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            expected = act.forward(np.array([x[i, j]]))[0]
            assert full[i, j] == pytest.approx(expected)


# ---------- ReLU ----------


def test_relu_known_values():
    out = ReLU().forward(np.array([-2.0, -0.5, 0.0, 0.5, 3.0]))
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.5, 3.0])


def test_relu_range(x):
    assert np.all(ReLU().forward(x) >= 0)


def test_relu_is_idempotent(x):
    relu = ReLU()
    once = relu.forward(x)
    np.testing.assert_array_equal(relu.forward(once), once)


# ---------- Sigmoid ----------


def test_sigmoid_known_values():
    out = Sigmoid().forward(np.array([0.0, np.log(3.0), -np.log(3.0)]))
    np.testing.assert_allclose(out, [0.5, 0.75, 0.25])


def test_sigmoid_range(x):
    out = Sigmoid().forward(10 * x)
    assert np.all(out > 0) and np.all(out < 1)


def test_sigmoid_symmetry(x):
    # sigma(-x) = 1 - sigma(x).
    s = Sigmoid()
    np.testing.assert_allclose(s.forward(-x), 1.0 - s.forward(x))


def test_sigmoid_integer_array():
    # integer arrays are treated as double arrays
    np.testing.assert_allclose(
        Sigmoid().forward(np.array([0, 1])), [0.5, (1 / (1 + np.exp(-1)))]
    )


def test_sigmoid_is_stable_for_extreme_inputs():
    extreme = np.array([-1000.0, -710.0, 0.0, 710.0, 1000.0])
    with np.errstate(
        over="raise", invalid="raise", divide="raise"
    ):  # any overflow/invalid becomes an error, but underflow passes
        out = Sigmoid().forward(extreme)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-300)


def test_sigmoid_is_monotonic():
    xs = np.linspace(-20, 20, 1001)
    out = Sigmoid().forward(xs)
    assert np.all(np.diff(out) >= 0)


# ---------- Tanh ----------


def test_tanh_known_values():
    out = Tanh().forward(np.array([0.0, 1.0, -1.0]))
    np.testing.assert_allclose(out, [0.0, np.tanh(1.0), -np.tanh(1.0)])


def test_tanh_range(x):
    out = Tanh().forward(10 * x)
    assert np.all(out >= -1) and np.all(out <= 1)


def test_tanh_is_odd(x):
    t = Tanh()
    np.testing.assert_allclose(t.forward(-x), -t.forward(x))


def test_tanh_relates_to_sigmoid(x):
    # tanh(x) = 2 * sigma(2x) - 1.
    np.testing.assert_allclose(
        Tanh().forward(x), 2.0 * Sigmoid().forward(2.0 * x) - 1.0, atol=1e-12
    )
