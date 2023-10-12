import numpy as np
from octo.measurement import PathIntegral
import pytest


def test_pathintegral():
    N = 100
    M = 50
    x = np.linspace(0, 2 * np.pi, M)
    y = np.ones(N)

    X, Y = np.meshgrid(x, y)
    field = np.sin(X)

    pathintegral = PathIntegral(N, M)
    pathintegral.add_path((0, M // 4), (N - 1, M // 4))
    pathintegral.add_path((N // 2, 0), (N // 2, M - 1))

    expected = np.array([np.sum(field[:, M // 4]), np.sum(field[N // 2, :])])
    actual = pathintegral(field)
    assert np.allclose(expected, actual)


def test_addingpaths():
    start = np.array([0, 0])
    end = np.array([5, 3])
    pathintegral = PathIntegral(100, 50, npaths=0)
    pathintegral.add_path(start, end)

    assert pathintegral.path_matrix.shape[0] == 1
    assert pathintegral.path_matrix.shape[1] == 100 * 50

    pathintegral.add_path(start, end)
    assert pathintegral.path_matrix.shape[0] == 2

    with pytest.raises(ValueError):
        pathintegral.add_path((100, 0), (0, 0))

    with pytest.raises(ValueError):
        pathintegral.add_path((0, 50), (0, 0))

    with pytest.raises(ValueError):
        pathintegral.add_path((0, 0), (100, 0))

    with pytest.raises(ValueError):
        pathintegral.add_path((0, 0), (0, 50))
