import numpy as np
from numpy.typing import NDArray


def make_xor(
    n_samples: int, noise: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate noisy XOR data.

    Samples cycle through the corners (0,0), (1,0), (0,1), (1,1), which have
    labels 0, 1, 1, 0, and Gaussian noise with standard deviation ``noise``
    is added to each coordinate. The data is not shuffled.

    Args:
        n_samples: Number of samples N. Must be positive.
        noise: Standard deviation of the noise. Must be >= 0 (0 gives
            exact corners).
        rng: Generator that draws the noise.

    Returns:
        x of shape (N, 2), float64, and y of shape (N,), int64, in {0, 1}.

    Raises:
        ValueError: If ``n_samples`` is not positive or ``noise`` negative.
    """
    if n_samples <= 0:
        raise ValueError("The number of samples must be positive.")

    if noise < 0:
        raise ValueError("Noise cannot be negative.")

    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels_corners = np.array([0, 1, 1, 0], dtype=np.int64)
    index_arr = np.arange(n_samples) % 4
    samples = corners[index_arr]
    labels = labels_corners[index_arr]
    samples += rng.normal(size=(n_samples, 2)) * noise
    return (samples, labels)


def make_blobs(
    n_samples: int,
    n_classes: int,
    std: float,
    rng: np.random.Generator,
    radius: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate Gaussian clouds centered on a circle.

    Class k has center radius * (cos(2*pi*k/K), sin(2*pi*k/K)) and samples
    are center plus std multiplied by Gaussian noise. Sample i gets class
    i modulo K, so classes are balanced up to one sample. The data is not shuffled.

    If ``std`` is large relative to the distance between neighboring centers
    the classes overlap and 100% accuracy is impossible.

    Args:
        n_samples: Number of samples N.
        n_classes: Number of classes K.
        std: Standard deviation of the noise.
        rng: Generator that draws the noise.
        radius: Radius of the circle of centers.

    Returns:
        x of shape (N, 2) and y of shape (N,).

    Raises:
        ValueError: If n_samples is not positive, n_classes is not at least 2,
        std is negative or radius is not positive.
    """
    if n_samples <= 0:
        raise ValueError("The number of samples must be positive.")

    if n_classes < 2:
        raise ValueError("The number of classes must be greater than 1.")

    if std < 0:
        raise ValueError("Standard deviation cannot be negative.")

    if radius <= 0:
        raise ValueError("The radius must be positive.")

    angles = 2 * np.pi * np.arange(n_classes) / n_classes
    centers = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    labels = np.arange(n_samples, dtype=np.int64) % n_classes
    samples = centers[labels]
    samples += rng.normal(size=(n_samples, 2)) * std
    return (samples, labels)
