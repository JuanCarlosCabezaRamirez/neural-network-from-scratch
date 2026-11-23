import numpy as np
import pytest

from numpy_nn.datasets import make_blobs, make_xor


def test_xor_shapes_and_dtypes():
    x, y = make_xor(40, 0.1, np.random.default_rng(0))
    assert x.shape == (40, 2) and y.shape == (40,)
    assert x.dtype == np.float64 and y.dtype == np.int64


def test_xor_without_noise_gives_exact_corners():
    x, y = make_xor(40, 0.0, np.random.default_rng(0))
    assert len(np.unique(x, axis=0)) == 4
    expected = np.logical_xor(x[:, 0] > 0.5, x[:, 1] > 0.5).astype(np.int64)
    np.testing.assert_array_equal(y, expected)


def test_xor_labels_follow_the_corner_with_small_noise():
    x, y = make_xor(200, 0.05, np.random.default_rng(0))
    expected = np.logical_xor(x[:, 0] > 0.5, x[:, 1] > 0.5).astype(np.int64)
    np.testing.assert_array_equal(y, expected)


def test_xor_classes_are_balanced():
    _, y = make_xor(41, 0.1, np.random.default_rng(0))
    counts = np.bincount(y, minlength=2)
    assert abs(counts[0] - counts[1]) <= 1 + 1  # corners are dealt in groups of 4


def test_blobs_shapes_and_dtypes():
    x, y = make_blobs(60, 3, 0.5, np.random.default_rng(0))
    assert x.shape == (60, 2) and y.shape == (60,)
    assert x.dtype == np.float64 and y.dtype == np.int64


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_blobs_labels_and_balance(n_classes):
    _, y = make_blobs(50, n_classes, 0.5, np.random.default_rng(0))
    counts = np.bincount(y, minlength=n_classes)
    assert set(np.unique(y)) == set(range(n_classes))
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_blobs_without_spread_gives_one_point_per_class(n_classes):
    x, y = make_blobs(50, n_classes, 0.0, np.random.default_rng(0))
    assert len(np.unique(x, axis=0)) == n_classes
    for k in range(n_classes):
        assert len(np.unique(x[y == k], axis=0)) == 1


def test_blobs_radius_scales_the_centers():
    small, _ = make_blobs(30, 3, 0.0, np.random.default_rng(0), radius=1.0)
    large, _ = make_blobs(30, 3, 0.0, np.random.default_rng(0), radius=4.0)
    np.testing.assert_allclose(large, 4.0 * small)


@pytest.mark.parametrize(
    "make", [lambda r: make_xor(30, 0.1, r), lambda r: make_blobs(30, 3, 0.5, r)]
)
def test_same_seed_same_data_different_seed_different_data(make):
    x1, y1 = make(np.random.default_rng(3))
    x2, y2 = make(np.random.default_rng(3))
    x3, _ = make(np.random.default_rng(4))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(x1, x3)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: make_xor(0, 0.1, r),
        lambda r: make_xor(-3, 0.1, r),
        lambda r: make_xor(10, -0.1, r),
        lambda r: make_blobs(0, 3, 0.5, r),
        lambda r: make_blobs(10, 1, 0.5, r),
        lambda r: make_blobs(10, 3, -0.5, r),
        lambda r: make_blobs(10, 3, 0.5, r, radius=0.0),
    ],
)
def test_invalid_arguments_raise(call):
    with pytest.raises(ValueError):
        call(np.random.default_rng(0))
