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
