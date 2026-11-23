import numpy as np
import pytest

from numpy_nn.metrics import accuracy


def test_known_value():
    logits = np.array(
        [
            [2.0, 1.0, 0.0],  # predicts 0
            [0.0, 3.0, 1.0],  # predicts 1
            [1.0, 0.0, 2.0],  # predicts 2
            [5.0, 0.0, 0.0],
        ]
    )  # predicts 0
    labels = np.array([0, 1, 0, 2])  # right, right, wrong, wrong
    assert accuracy(logits, labels) == pytest.approx(0.5)


def test_all_correct_is_one():
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 1, 1])
    assert accuracy(logits, labels) == 1.0


def test_all_wrong_is_zero():
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([1, 0, 0])
    assert accuracy(logits, labels) == 0.0


def test_returns_python_float():
    result = accuracy(np.array([[1.0, 0.0]]), np.array([0]))
    assert isinstance(result, float)


def test_invariant_to_adding_a_constant_per_row():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(8, 4))
    labels = rng.integers(0, 4, size=8)
    shifts = rng.normal(size=(8, 1)) * 100.0
    assert accuracy(logits + shifts, labels) == accuracy(logits, labels)


def test_ties_resolve_to_first_class():
    logits = np.zeros((4, 3))
    labels = np.array([0, 0, 1, 2])
    assert accuracy(logits, labels) == pytest.approx(0.5)


def test_binary_with_two_columns():
    logits = np.array([[0.2, 0.8], [0.9, 0.1]])
    assert accuracy(logits, np.array([1, 0])) == 1.0


def test_inputs_not_mutated():
    logits = np.array([[2.0, 1.0], [0.0, 3.0]])
    labels = np.array([0, 1])
    logits_copy, labels_copy = logits.copy(), labels.copy()
    accuracy(logits, labels)
    np.testing.assert_array_equal(logits, logits_copy)
    np.testing.assert_array_equal(labels, labels_copy)


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (np.zeros(3), np.array([0, 1, 2])),  # logits not 2D
        (np.zeros((0, 3)), np.array([], dtype=int)),  # empty batch
        (np.zeros((3, 2)), np.array([0, 1])),  # wrong length
        (np.zeros((3, 2)), np.zeros((3, 1), dtype=int)),  # wrong ndim
        (np.zeros((3, 2)), np.array([0.0, 1.0, 0.0])),  # float labels
    ],
)
def test_invalid_inputs_raise(logits, labels):
    with pytest.raises(ValueError):
        accuracy(logits, labels)
