import numpy as np


class PathIntegral:
    def __init__(
        self, Nx: int, Ny: int, path_matrix: np.ndarray = None, npaths: int = 0
    ) -> None:
        self.Nx = Nx
        self.Ny = Ny
        if path_matrix is None:
            if npaths != -1:  # assume we want a random path matrix
                self.path_matrix = self.create_random_paths(npaths)
            else:  # create an empty path matrix
                self.path_matrix = np.zeros((1, self.Nx * self.Ny))
        else:  # clip path matrix to npaths if given
            self.path_matrix = path_matrix[:npaths]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        X is a 2D array of shape (Nx, Ny) representing the field
        over which we are integrating
        """
        return self.path_matrix @ X.ravel()

    def plot(self):
        import matplotlib.pyplot as plt

        pm = self.path_matrix.sum(axis=0).reshape(self.Nx, self.Ny)
        plt.imshow(pm)
        plt.show()

    def add_path(self, start, end):
        """
        adds a path to the path matrix
        start and end are cooridinates in the reference frame of the array
        i.e. start = (0,0) is the top left corner and end = (Nx-1, Ny-1) is the bottom right corner
        """
        if np.any(np.array([start[0], end[0]]) >= self.Nx):
            raise ValueError("startx and endx must be less than Nx")
        if np.any(np.array([start[1], end[1]]) >= self.Ny):
            raise ValueError("starty and endy must be less than Ny")

        if self.path_matrix.shape[0] == 1 and np.all(
            self.path_matrix == 0
        ):  # empty path matrix
            self.path_matrix[0] = self.sample_path(start, end).ravel()
        else:
            self.path_matrix = np.vstack(
                (self.path_matrix, self.sample_path(start, end).ravel())
            )

    def sample_path(self, start, end):
        """
        finds points along a path
        """
        startx, starty = start
        endx, endy = end
        path = np.zeros((self.Nx, self.Ny))
        if startx == endx:  # vertical line
            path[startx, np.arange(starty, endy + 1, step=np.sign(endy - starty))] = 1
            return path
        x = np.arange(startx, endx + 1, step=np.sign(endx - startx))
        y = self._line_from_points((startx, starty), (endx, endy))(x)
        path[x, np.round(y).astype(int)] = 1
        return path

    def create_random_paths(self, npaths: int) -> np.ndarray:
        path_matrix = np.zeros((npaths, self.Nx * self.Ny))
        for i in range(npaths):
            path_matrix[i, :] = self._random_path()
        return path_matrix

    def _random_path(self) -> np.ndarray:
        # path should start outside the box, so either startx or starty
        # needs to be 0
        startx = np.random.choice([0, np.random.randint(1, self.Nx)])
        starty = 0 if startx != 0 else np.random.randint(1, self.Ny)

        # path should end outside the box, so either endx or endy
        # needs to be N-1
        # while loop to ensure that start and end are not the same
        endx, endy = startx, starty
        while endx == startx and endy == starty:
            endx = np.random.choice([self.Nx - 1, np.random.randint(0, self.Nx - 1)])
            endy = (
                self.Ny - 1
                if endx != self.Nx - 1
                else np.random.randint(0, self.Ny - 1)
            )

        return self.sample_path((startx, starty), (endx, endy)).ravel()

    def _line_from_points(self, start, end):
        """
        Given two points, return the line that connects them
        """
        x1, y1 = start
        x2, y2 = end
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return lambda x: slope * x + intercept
