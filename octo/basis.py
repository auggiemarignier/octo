import numpy as np
from scipy.fft import idct
from typing import Callable


class BaseBasis:
    def __init__(self, N: int) -> None:
        """
        N: number of basis functions
        """
        self.N = N
        self.jacobian = None

    def compute_jacobian(self, forward: Callable) -> None:
        """
        forward is the measurment operator.
        It should take a vector of length N or an array of shape (N, N).
        """
        self.jacobian = forward(self.basis)

    def _create_basis(self) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self, N: int, method="idct") -> None:
        super().__init__(N)

        if method not in ["idct", "cos"]:
            raise ValueError("method must be one of 'idct', 'cos'")
        self.method = (
            self._basis_from_idct if method == "idct" else self._basis_from_cos
        )

        self._create_basis()

    def plot(self):
        import matplotlib.pyplot as plt

        self._create_basis(_resolution=10 * self.N)
        for i, basis in enumerate(self.basis):
            plt.plot(i + basis)
        plt.show()

        self._create_basis()  # reset basis

    def _create_basis(self, _resolution: int = None) -> None:
        """
        The _resolution argument is included for testing purposes. It is not
        intended to be used by the user. It represents the number of points in
        the range [0, pi) that the basis functions are evaluated at.
        """
        if _resolution is None:
            _resolution = self.N
        self.basis = np.zeros((self.N, _resolution))
        for i in range(self.N):
            self.basis[i, :] = self.method(i, _resolution)

    def _basis_from_idct(self, i, _resolution) -> np.ndarray:
        return idct(np.eye(_resolution)[i, :], norm="ortho")

    def _basis_from_cos(self, i, _resolution) -> np.ndarray:
        norm = np.sqrt(_resolution / 2)
        if i == 0:
            norm *= np.sqrt(2)
        n = 2 * np.arange(_resolution) + 1
        return np.cos(i * np.pi * n / (2 * _resolution)) / norm


class CosineBasis2D(BaseBasis):
    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self._create_basis()

    def plot(self):
        import matplotlib.pyplot as plt

        factor = 10  # upsampling factor

        Bx = CosineBasis(self.Nx)
        Bx._create_basis(_resolution=factor * self.Nx)
        By = CosineBasis(self.Ny)
        By._create_basis(_resolution=factor * self.Ny)

        basis_matrix = np.outer(Bx.basis, By.basis)
        plt.imshow(basis_matrix, cmap="RdBu")
        for x in range(1, self.Ny):
            plt.axvline(x * factor * self.Ny, color="k", ls="--")
        for y in range(1, self.Nx):
            plt.axhline(y * factor * self.Nx, color="k", ls="--")
        plt.axis(False)
        plt.show()

    def _create_basis(self) -> None:
        Bx = CosineBasis(self.Nx)
        By = CosineBasis(self.Ny)
        self.basis = _unravel(np.outer(Bx.basis, By.basis), self.Nx, self.Ny)


class PixelBasis(BaseBasis):
    def __init__(self, N: int) -> None:
        super().__init__(N)
        self._create_basis()

    def plot(self):
        import matplotlib.pyplot as plt

        x_fine = np.linspace(0, self.N, 1000)
        for i in range(self.N):
            basis_fine = np.zeros_like(x_fine)
            basis_fine[np.argmin(np.abs(x_fine - i))] = 0.95
            plt.plot(x_fine, i + basis_fine)

        plt.show()

    def _create_basis(self) -> None:
        self.basis = np.eye(self.N)


class PixelBasis2D(BaseBasis):
    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self._create_basis()

    def plot(self):
        import matplotlib.pyplot as plt

        Bx = PixelBasis(self.Nx)
        By = PixelBasis(self.Ny)
        basis_matrix = np.outer(Bx.basis, By.basis)

        plt.imshow(basis_matrix, cmap="binary")
        for x in range(1, self.Ny):
            plt.axvline(x * self.Ny, color="k", ls="--")
        for y in range(1, self.Nx):
            plt.axhline(y * self.Nx, color="k", ls="--")
        plt.axis(False)
        plt.show()

    def _create_basis(self) -> None:
        Bx = PixelBasis(self.Nx)
        By = PixelBasis(self.Ny)
        self.basis = _unravel(np.outer(Bx.basis, By.basis), self.Nx, self.Ny)


def _unravel(basis_matrix: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Combining 2 1D basis classes by taking the outer product of their basis
    creates a matrix of 2D basis functions. This function unravels that matrix
    such that the 2D basis functions are flattened into a row vector.
    """
    unraveled = np.zeros((nx * ny, nx * ny))
    for i in range(nx):  # row
        for j in range(ny):  # column
            k = i * ny + j
            unraveled[k, :] = basis_matrix[
                i * nx : (i + 1) * nx, j * ny : (j + 1) * ny
            ].flatten()
    return unraveled


def _reravel(basis_matrix: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    The inverse of _unravel
    """
    reraveled = np.zeros((nx * nx, ny * ny))
    for i in range(nx):  # row
        for j in range(ny):  # column
            k = i * ny + j
            reraveled[i * nx : (i + 1) * nx, j * ny : (j + 1) * ny] = basis_matrix[
                k, :
            ].reshape((nx, ny))
    return reraveled
