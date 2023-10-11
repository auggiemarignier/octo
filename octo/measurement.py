import numpy as np


class PathIntegral:
    def __init__(
        self, N: int, path_matrix: np.ndarray = None, npaths: int = -1
    ) -> None:
        self.N = N
        if path_matrix is None:
            self.path_matrix = self.create_random_paths(npaths)
        else:  # clip path matrix to npaths if given
            self.path_matrix = path_matrix[:npaths]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        X is a 2D array of shape (N, N) representing the field
        over which we are integrating
        """
        return self.path_matrix @ X

    def plot(self):
        import matplotlib.pyplot as plt

        pm = self.path_matrix.sum(axis=0).reshape(self.N, self.N)
        plt.imshow(pm)
        plt.show()

    def create_random_paths(self, npaths: int) -> np.ndarray:
        path_matrix = np.zeros((npaths, self.N * self.N))
        for i in range(npaths):
            path_matrix[i, :] = self._random_path()
        return path_matrix

    def _random_path(self) -> np.ndarray:
        path = np.zeros((self.N, self.N))

        # path should start outside the box, so either startx or starty
        # needs to be 0
        startx = np.random.choice([0, np.random.randint(1, self.N)])
        starty = 0 if startx != 0 else np.random.randint(1, self.N)

        # path should end outside the box, so either endx or endy
        # needs to be N-1
        # while loop to ensure that start and end are not the same
        endx, endy = startx, starty
        while endx == startx and endy == starty:
            endx = np.random.choice([self.N - 1, np.random.randint(0, self.N - 1)])
            endy = (
                self.N - 1 if endx != self.N - 1 else np.random.randint(0, self.N - 1)
            )

        x = np.arange(startx, endx + 1, step=np.sign(endx - startx))
        y = self._line_from_points((startx, starty), (endx, endy))(x)
        path[x, y.astype(int)] = 1
        return path.ravel()

    def _line_from_points(self, start, end):
        """
        Given two points, return the line that connects them
        """
        x1, y1 = start
        x2, y2 = end
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return lambda x: slope * x + intercept
