"""Train a Dense -> Tanh -> Dense network on noisy Gaussian clouds centered on a circle."""

import numpy as np

from numpy_nn.activations import Tanh
from numpy_nn.datasets import make_blobs
from numpy_nn.layers import Dense
from numpy_nn.losses import SoftmaxCrossEntropy
from numpy_nn.metrics import accuracy
from numpy_nn.optimizers import SGD
from numpy_nn.sequential import Sequential
from numpy_nn.training import train

# Untuned starting values
DATA_SEED = 0
TEST_SEED = 1
INIT_SEED = 2
SHUFFLE_SEED = 3
N_TRAIN = 300
N_CLASSES = 3
N_TEST = 300
STD = 0.8
RADIUS = 3.0
HIDDEN = 8
LR = 0.1
EPOCHS = 300
BATCH_SIZE = 20


def main() -> None:
    data_rng = np.random.default_rng(DATA_SEED)
    test_rng = np.random.default_rng(TEST_SEED)
    init_rng = np.random.default_rng(INIT_SEED)
    shuffle_rng = np.random.default_rng(SHUFFLE_SEED)
    x_train, y_train = make_blobs(N_TRAIN, N_CLASSES, STD, data_rng, RADIUS)
    x_test, y_test = make_blobs(N_TEST, N_CLASSES, STD, test_rng, RADIUS)
    model = Sequential(
        [Dense(2, HIDDEN, init_rng), Tanh(), Dense(HIDDEN, N_CLASSES, init_rng)]
    )
    loss = SoftmaxCrossEntropy()
    optimizer = SGD(LR)
    history = train(
        model, loss, optimizer, x_train, y_train, EPOCHS, BATCH_SIZE, shuffle_rng
    )
    print(f"First epoch's loss: {history[0]:.4f}, last epoch's loss: {history[-1]:.4f}")
    print(f"Epoch's loss history: {[f'{i:.4f}' for i in history[:: EPOCHS // 10]]}")
    print(
        f"Accuracy of the model in training: {accuracy(model.forward(x_train), y_train)}"
    )
    print(
        f"Accuracy of the model in testing: {accuracy(model.forward(x_test), y_test)}"
    )


if __name__ == "__main__":
    main()
