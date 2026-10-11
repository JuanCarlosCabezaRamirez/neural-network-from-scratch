import numpy_nn


def test_version():
    assert hasattr(numpy_nn, "__version__")
    assert isinstance(numpy_nn.__version__, str)
    assert numpy_nn.__version__ == "0.1.0"
