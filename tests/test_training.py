import numpy as np
import pytest

from numpy_nn.activations import Tanh
from numpy_nn.datasets import make_xor
from numpy_nn.layers import Dense
from numpy_nn.losses import SoftmaxCrossEntropy
from numpy_nn.metrics import accuracy
from numpy_nn.optimizers import SGD
from numpy_nn.sequential import Sequential
from numpy_nn.training import train


def make_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    return x, y


def make_model(seed=0):
    rng = np.random.default_rng(seed)
    return Sequential([Dense(2, 8, rng), Tanh(), Dense(8, 2, rng)])


class CountingSGD(SGD):
    def __init__(self, lr):
        super().__init__(lr)
        self.calls = 0

    def step(self, parameters):
        self.calls += 1
        super().step(parameters)


def test_one_loss_per_epoch():
    x, y = make_data()
    history = train(
        make_model(),
        SoftmaxCrossEntropy(),
        SGD(0.1),
        x,
        y,
        epochs=5,
        batch_size=8,
        rng=np.random.default_rng(0),
    )
    assert len(history) == 5
    assert all(isinstance(value, float) for value in history)


def test_one_update_per_batch():
    x, y = make_data(n=10)
    optimizer = CountingSGD(0.1)
    train(
        make_model(),
        SoftmaxCrossEntropy(),
        optimizer,
        x,
        y,
        epochs=3,
        batch_size=4,
        rng=np.random.default_rng(0),
    )
    assert optimizer.calls == 3 * 3  # ceil(10 / 4) = 3 batches per epoch


def test_epoch_loss_is_sample_weighted():
    # With a negligible learning rate the weights do not change, so every
    # batch is evaluated at the same weights and the weighted epoch loss
    # must equal the loss on the whole dataset.
    x, y = make_data(n=10)
    model = make_model()
    expected = SoftmaxCrossEntropy().forward(model.forward(x), y)
    history = train(
        model,
        SoftmaxCrossEntropy(),
        SGD(1e-300),
        x,
        y,
        epochs=2,
        batch_size=4,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(history, [expected, expected], rtol=1e-9)


def test_loss_decreases_on_separable_problem():
    x, y = make_data()
    model = make_model()
    loss = SoftmaxCrossEntropy()
    before = loss.forward(model.forward(x), y)
    history = train(
        model,
        SoftmaxCrossEntropy(),
        SGD(0.1),
        x,
        y,
        epochs=30,
        batch_size=8,
        rng=np.random.default_rng(0),
    )
    after = loss.forward(model.forward(x), y)
    assert history[-1] < history[0]
    assert after < before


def test_same_seed_same_history():
    x, y = make_data()
    histories = [
        train(
            make_model(1),
            SoftmaxCrossEntropy(),
            SGD(0.1),
            x,
            y,
            epochs=5,
            batch_size=8,
            rng=np.random.default_rng(7),
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(histories[0], histories[1])


def test_different_shuffle_seed_different_history():
    x, y = make_data()
    first, second = (
        train(
            make_model(1),
            SoftmaxCrossEntropy(),
            SGD(0.1),
            x,
            y,
            epochs=5,
            batch_size=8,
            rng=np.random.default_rng(seed),
        )
        for seed in (7, 8)
    )
    assert not np.array_equal(first, second)


def test_dataset_not_modified():
    x, y = make_data()
    x_copy, y_copy = x.copy(), y.copy()
    train(
        make_model(),
        SoftmaxCrossEntropy(),
        SGD(0.1),
        x,
        y,
        epochs=3,
        batch_size=8,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_parameters_are_updated():
    x, y = make_data()
    model = make_model()
    weights_before = model.parameters()[0][0].copy()
    train(
        model,
        SoftmaxCrossEntropy(),
        SGD(0.1),
        x,
        y,
        epochs=1,
        batch_size=8,
        rng=np.random.default_rng(0),
    )
    assert not np.array_equal(model.parameters()[0][0], weights_before)


@pytest.mark.parametrize(
    ("epochs", "batch_size", "x_rows", "y_rows"),
    [(0, 8, 40, 40), (-1, 8, 40, 40), (3, 0, 40, 40), (3, 8, 40, 39)],
)
def test_invalid_arguments_raise(epochs, batch_size, x_rows, y_rows):
    x, y = make_data(n=40)
    with pytest.raises(ValueError):
        train(
            make_model(),
            SoftmaxCrossEntropy(),
            SGD(0.1),
            x[:x_rows],
            y[:y_rows],
            epochs=epochs,
            batch_size=batch_size,
            rng=np.random.default_rng(0),
        )


def test_x_must_be_2d():
    _, y = make_data()
    with pytest.raises(ValueError):
        train(
            make_model(),
            SoftmaxCrossEntropy(),
            SGD(0.1),
            np.zeros(40),
            y,
            epochs=1,
            batch_size=8,
            rng=np.random.default_rng(0),
        )


def test_learns_xor():
    x, y = make_xor(200, 0.1, np.random.default_rng(0))
    model = make_model(2)  # Dense(2, 8) -> Tanh -> Dense(8, 2)
    train(
        model,
        SoftmaxCrossEntropy(),
        SGD(0.1),
        x,
        y,
        epochs=300,
        batch_size=20,
        rng=np.random.default_rng(3),
    )
    assert accuracy(model.forward(x), y) > 0.9
