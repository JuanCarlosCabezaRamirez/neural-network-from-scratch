import numpy as np
import pytest

from numpy_nn.data import make_batches

CASES = [(10, 3), (10, 5), (10, 10), (10, 1), (7, 20), (1, 1), (20, 4)]


@pytest.mark.parametrize(("n", "b"), CASES)
def test_every_sample_appears_exactly_once(n, b):
    batches = make_batches(n, b, np.random.default_rng(0))
    all_indices = np.concatenate(batches)
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(n))


@pytest.mark.parametrize(("n", "b"), CASES)
def test_batch_sizes(n, b):
    batches = make_batches(n, b, np.random.default_rng(0))
    expected = [b] * (n // b) + ([n % b] if n % b else [])
    if b > n:
        expected = [n]
    assert [len(batch) for batch in batches] == expected


def test_indices_are_integers():
    batches = make_batches(10, 3, np.random.default_rng(0))
    assert all(np.issubdtype(batch.dtype, np.integer) for batch in batches)


def test_indices_are_shuffled():
    batches = make_batches(20, 5, np.random.default_rng(0))
    assert not np.array_equal(np.concatenate(batches), np.arange(20))


def test_same_seed_same_batches():
    first = make_batches(20, 6, np.random.default_rng(42))
    second = make_batches(20, 6, np.random.default_rng(42))
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_different_seed_different_order():
    first = np.concatenate(make_batches(20, 6, np.random.default_rng(0)))
    second = np.concatenate(make_batches(20, 6, np.random.default_rng(1)))
    assert not np.array_equal(first, second)


def test_order_changes_between_epochs():
    rng = np.random.default_rng(0)
    epoch_1 = np.concatenate(make_batches(20, 6, rng))
    epoch_2 = np.concatenate(make_batches(20, 6, rng))
    assert not np.array_equal(epoch_1, epoch_2)


@pytest.mark.parametrize(("n", "b"), [(0, 3), (-5, 3), (10, 0), (10, -1)])
def test_invalid_arguments_raise(n, b):
    with pytest.raises(ValueError):
        make_batches(n, b, np.random.default_rng(0))
