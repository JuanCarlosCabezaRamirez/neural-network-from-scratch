import numpy as np
import pytest
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose

from numpy_nn.activations import ReLU, Sigmoid, Tanh

ACTIVATIONS = [ReLU, Sigmoid, Tanh]


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
@pytest.mark.parametrize("shape", [(3,), (4, 5), (2, 3, 4)])
def test_gradient_check(act_cls, shape):
    rng = np.random.default_rng(0)
    act = act_cls()
    # |x| >= 0.2 keeps us away from ReLU's kink at 0 (eps is 1e-6).
    x = rng.uniform(0.2, 2.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)
    g = rng.normal(size=shape)

    def loss() -> float:
        return float(np.sum(g * act.forward(x)))

    loss()
    dx = act.backward(g)
    assert_allclose(dx, numerical_gradient(loss, x), rtol=1e-6, atol=1e-8)


def test_relu_backward_known_values():
    act = ReLU()
    act.forward(np.array([-1.0, 0.0, 2.0]))
    assert_allclose(act.backward(np.array([5.0, 5.0, 5.0])), [0.0, 0.0, 5.0])


def test_sigmoid_backward_at_zero():
    act = Sigmoid()
    act.forward(np.array([0.0]))
    assert_allclose(act.backward(np.array([4.0])), [1.0])  # 4 * 0.25


def test_tanh_backward_at_zero():
    act = Tanh()
    act.forward(np.array([0.0]))
    assert_allclose(act.backward(np.array([3.0])), [3.0])  # 3 * 1


@pytest.mark.parametrize("act_cls", [Sigmoid, Tanh])
def test_saturated_gradients_are_finite(act_cls):
    act = act_cls()
    act.forward(np.array([-1000.0, -710.0, 710.0, 1000.0]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dx = act.backward(np.ones(4))
    assert np.all(np.isfinite(dx))
    assert_allclose(dx, 0.0, atol=1e-12)


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_backward_does_not_mutate_dy(act_cls):
    act = act_cls()
    act.forward(np.array([[-1.0, 0.5], [2.0, -0.3]]))
    dy = np.array([[1.0, 2.0], [3.0, 4.0]])
    dy_copy = dy.copy()
    act.backward(dy)
    assert_allclose(dy, dy_copy, rtol=0, atol=0)


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_backward_before_forward_raises(act_cls):
    with pytest.raises(RuntimeError):
        act_cls().backward(np.ones(3))


@pytest.mark.parametrize("act_cls", ACTIVATIONS)
def test_backward_rejects_wrong_dy_shape(act_cls):
    act = act_cls()
    act.forward(np.ones((2, 3)))
    with pytest.raises(ValueError):
        act.backward(np.ones((3, 2)))
