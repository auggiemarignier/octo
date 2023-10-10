from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.fft import idct


class BaseBasis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, N: int) -> None:
        """
        N: number of basis functions
        """
        self.N = N
        self.jacobian = None

    @abstractmethod
    def compute_jacobian(self, forward) -> None:
        pass

    @abstractmethod
    def _create_basis(self) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self, N: int, resolution: int = None, method="idct") -> None:
        super().__init__(N)
        self.resolution = (
            N if resolution is None else resolution
        )  # number of points per cycle [0, pi)

        if method not in ["idct", "cos"]:
            raise ValueError("method must be one of 'idct', 'cos'")
        self.method = (
            self._basis_from_idct if method == "idct" else self._basis_from_cos
        )

        self._create_basis()

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian

    def _create_basis(self) -> None:
        self.basis = np.zeros((self.N, self.resolution))
        for i in range(self.N):
            self.basis[i, :] = self.method(i)

    def _basis_from_idct(self, i) -> np.ndarray:
        return idct(np.eye(self.resolution)[i, :], norm="ortho")

    def _basis_from_cos(self, i) -> np.ndarray:
        norm = np.sqrt(self.resolution / 2)
        if i == 0:
            norm *= np.sqrt(2)
        n = 2 * np.arange(self.resolution) + 1
        return np.cos(i * np.pi * n / (2 * self.resolution)) / norm


class CosineBasis2D(BaseBasis):
    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self._create_basis()

    def compute_jacobian(self, forward) -> None:
        pass

    def _create_basis(self) -> None:
        Bx = CosineBasis(self.Nx)
        By = CosineBasis(self.Ny)
        self.basis = _unravel(np.outer(Bx.basis, By.basis), self.Nx, self.Ny)


class PixelBasis(BaseBasis):
    def __init__(self, N: int) -> None:
        super().__init__(N)
        self._create_basis()

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian

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

    def compute_jacobian(self, forward) -> None:
        pass

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
