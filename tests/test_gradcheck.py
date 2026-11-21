import numpy as np
from gradcheck import numerical_gradient
from numpy.testing import assert_allclose


def test_matches_known_gradient_of_sum_of_squares():
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    grad = numerical_gradient(lambda: np.sum(x**2), x)
    assert_allclose(grad, 2 * x, rtol=1e-6)


def test_restores_the_array():
    x = np.array([1.0, 2.0, 3.0])
    original = x.copy()
    numerical_gradient(lambda: np.sum(x**3), x)
    assert_allclose(x, original, rtol=0, atol=0)
