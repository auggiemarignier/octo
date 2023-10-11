import numpy as np
from octo.measurement import PathIntegral


def test_pathintegral():
    import matplotlib.pyplot as plt

    N = 100
    M = 50

    x = np.linspace(0, 2 * np.pi, M)
    y = np.ones(N)

    X, Y = np.meshgrid(x, y)
    field = np.sin(X)
    plt.imshow(field)
    plt.show()

    pathintegral = PathIntegral(N, M)
    pathintegral.add_path((0, M // 4), (N - 1, M // 4))
    pathintegral.plot()

    expected = np.array([np.sum(field[:, M // 4])])
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
