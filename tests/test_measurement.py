import numpy as np
from octo.measurement import PathIntegral


def test_pathintegral():
    def field(pos):
        x, y = pos
        return np.sin(x) * np.cos(y)

    start = np.array([0, 0])
    end = np.array([np.pi / 2, np.pi / 2])
    num_steps = 100

    pathintegral = PathIntegral(100, 50, npaths=5)
    pathintegral.plot()


def test_addingpaths():
    start = np.array([0, 0])
    end = np.array([5, 3])
    pathintegral = PathIntegral(100, 50, npaths=0)
    pathintegral.add_path(start, end)

    assert pathintegral.path_matrix.shape[0] == 1
    assert pathintegral.path_matrix.shape[1] == 100 * 50

    pathintegral.add_path(start, end)
    assert pathintegral.path_matrix.shape[0] == 2
