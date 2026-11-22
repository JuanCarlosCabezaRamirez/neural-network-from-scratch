"""Tests for the loss functions: MSELoss and SoftmaxCrossEntropy."""

import numpy as np
import pytest
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose

from numpy_nn.losses import MSELoss, SoftmaxCrossEntropy

RTOL, ATOL = 1e-6, 1e-8


# ---------------------------------------------------------------- MSELoss


def test_mse_known_values():
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[1.0, 0.0], [1.0, 4.0]])
    loss_fn = MSELoss()

    loss = loss_fn.forward(y_pred, y_true)

    # differences [[0, 2], [2, 0]] -> squares sum to 8 -> mean over 4 elements
    assert isinstance(loss, float)
    assert_allclose(loss, 2.0)
    assert_allclose(loss_fn.backward(), [[0.0, 1.0], [1.0, 0.0]])  # 2*diff/4


def test_mse_zero_for_perfect_predictions():
    x = np.random.default_rng(0).normal(size=(4, 3))
    loss_fn = MSELoss()
    assert loss_fn.forward(x, x.copy()) == 0.0
    assert_allclose(loss_fn.backward(), np.zeros((4, 3)))


@pytest.mark.parametrize("shape", [(3,), (4, 2), (2, 3, 4)])
def test_mse_gradient_shape_and_finite_difference(shape):
    rng = np.random.default_rng(1)
    y_pred = rng.normal(size=shape)
    y_true = rng.normal(size=shape)
    loss_fn = MSELoss()

    loss_fn.forward(y_pred, y_true)
    analytic = (
        loss_fn.backward()
    )  # analytic first: the numerical estimate re-runs forward
    numeric = numerical_gradient(lambda: loss_fn.forward(y_pred, y_true), y_pred)

    assert analytic.shape == y_pred.shape
    assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_mse_empty_input_raises():
    with pytest.raises(ValueError):
        MSELoss().forward(np.zeros((0, 3)), np.zeros((0, 3)))


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="(?i)shape"):
        MSELoss().forward(np.zeros((4, 2)), np.zeros((4, 3)))


# ---------------------------------------------------- SoftmaxCrossEntropy


def naive_cross_entropy(logits, labels):
    """Reference implementation, only valid for moderate logits."""
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(labels)), labels]))


def test_ce_uniform_logits_known_values():
    logits = np.zeros((2, 4))
    labels = np.array([0, 3])
    loss_fn = SoftmaxCrossEntropy()

    loss = loss_fn.forward(logits, labels)

    assert isinstance(loss, float)
    assert_allclose(loss, np.log(4))
    # (p - onehot) / N with p = 0.25 and N = 2
    expected = np.array([[-0.375, 0.125, 0.125, 0.125], [0.125, 0.125, 0.125, -0.375]])
    assert_allclose(loss_fn.backward(), expected)


def test_ce_matches_naive_reference_on_moderate_logits():
    rng = np.random.default_rng(2)
    logits = rng.normal(scale=2.0, size=(6, 5))
    labels = rng.integers(0, 5, size=6)
    assert_allclose(
        SoftmaxCrossEntropy().forward(logits, labels),
        naive_cross_entropy(logits, labels),
    )


def test_ce_finite_difference():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(5, 4))
    labels = rng.integers(0, 4, size=5)
    loss_fn = SoftmaxCrossEntropy()

    loss_fn.forward(logits, labels)
    analytic = loss_fn.backward()
    numeric = numerical_gradient(lambda: loss_fn.forward(logits, labels), logits)

    assert analytic.shape == logits.shape
    assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_ce_gradient_rows_sum_to_zero():
    # Each row is (p - onehot)/N and both p and onehot sum to 1.
    rng = np.random.default_rng(4)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(rng.normal(size=(7, 3)), rng.integers(0, 3, size=7))
    assert_allclose(loss_fn.backward().sum(axis=1), np.zeros(7), atol=1e-12)


def test_ce_invariant_to_constant_shift_of_logits():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 3))
    labels = rng.integers(0, 3, size=4)
    a, b = SoftmaxCrossEntropy(), SoftmaxCrossEntropy()

    assert_allclose(a.forward(logits, labels), b.forward(logits + 100.0, labels))
    assert_allclose(a.backward(), b.backward(), atol=1e-12)


def test_ce_confident_correct_prediction_has_near_zero_loss():
    labels = np.array([0, 2, 1])
    logits = 50.0 * np.eye(3)[labels]
    loss_fn = SoftmaxCrossEntropy()
    assert 0.0 <= loss_fn.forward(logits, labels) < 1e-10
    assert_allclose(loss_fn.backward(), np.zeros((3, 3)), atol=1e-10)


def test_ce_stable_on_extreme_logits():
    logits = np.array(
        [[-1000.0, 0.0, 1000.0], [710.0, 710.0, 710.0], [1000.0, -1000.0, 0.0]]
    )
    labels = np.array([0, 1, 2])
    loss_fn = SoftmaxCrossEntropy()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = loss_fn.forward(logits, labels)
        grad = loss_fn.backward()

    # Row losses are 2000, log(3) and 1000.
    assert_allclose(loss, (2000.0 + np.log(3.0) + 1000.0) / 3.0)
    assert np.all(np.isfinite(grad))
    expected = np.array(
        [[-1 / 3, 0.0, 1 / 3], [1 / 9, -2 / 9, 1 / 9], [1 / 3, 0.0, -1 / 3]]
    )
    assert_allclose(grad, expected, atol=1e-12)


# --------------------------------------- properties shared by both losses


@pytest.mark.parametrize("make_case", ["mse", "ce"])
def test_loss_is_a_mean_over_the_batch(make_case):
    """Duplicating the batch keeps the loss and halves each per-sample gradient.

    This is the 1/N factor: it lives in the loss, never in the layers.
    """
    rng = np.random.default_rng(6)
    if make_case == "mse":
        loss_fn, x, t = MSELoss(), rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    else:
        loss_fn, x, t = (
            SoftmaxCrossEntropy(),
            rng.normal(size=(5, 3)),
            rng.integers(0, 3, size=5),
        )

    loss_single = loss_fn.forward(x, t)
    grad_single = loss_fn.backward()
    loss_double = loss_fn.forward(np.concatenate([x, x]), np.concatenate([t, t]))
    grad_double = loss_fn.backward()

    assert_allclose(loss_double, loss_single)
    assert_allclose(grad_double[:5], grad_single / 2)
    assert_allclose(grad_double[5:], grad_single / 2)


@pytest.mark.parametrize("make_case", ["mse", "ce"])
def test_backward_is_repeatable_and_returns_a_fresh_array(make_case):
    rng = np.random.default_rng(7)
    if make_case == "mse":
        loss_fn, x, t = MSELoss(), rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    else:
        loss_fn, x, t = (
            SoftmaxCrossEntropy(),
            rng.normal(size=(4, 3)),
            rng.integers(0, 3, size=4),
        )

    loss_fn.forward(x, t)
    first = loss_fn.backward()
    expected = first.copy()
    first[...] = 123.0  # a caller scribbling on the result must not corrupt the cache

    assert_allclose(loss_fn.backward(), expected)


@pytest.mark.parametrize("make_case", ["mse", "ce"])
def test_inputs_are_not_mutated(make_case):
    rng = np.random.default_rng(8)
    if make_case == "mse":
        loss_fn, x, t = MSELoss(), rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    else:
        loss_fn, x, t = (
            SoftmaxCrossEntropy(),
            rng.normal(size=(4, 3)),
            rng.integers(0, 3, size=4),
        )
    x_before, t_before = x.copy(), t.copy()

    loss_fn.forward(x, t)
    loss_fn.backward()

    assert np.array_equal(x, x_before)
    assert np.array_equal(t, t_before)


@pytest.mark.parametrize("loss_cls", [MSELoss, SoftmaxCrossEntropy])
def test_backward_before_forward_raises(loss_cls):
    with pytest.raises(RuntimeError):
        loss_cls().backward()


# ----------------------------------- SoftmaxCrossEntropy input validation


def test_ce_logits_must_be_2d():
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy().forward(np.zeros(3), np.array([0]))


def test_ce_empty_batch_raises():
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy().forward(np.zeros((0, 3)), np.array([], dtype=int))


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 1]),  # wrong length for a batch of 3
        np.array([[0], [1], [2]]),  # wrong ndim
        np.array([0.0, 1.0, 2.0]),  # floats are not class indices
        np.array([0, 1, 3]),  # 3 is out of range for 3 classes
        np.array([0, 1, -1]),  # -1 would silently wrap to the last class
    ],
)
def test_ce_invalid_labels_raise(labels):
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy().forward(np.zeros((3, 3)), labels)
