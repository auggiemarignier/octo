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
    def create_basis(self) -> None:
        pass

    @abstractmethod
    def compute_jacobian(self, forward) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self, N: int, resolution: int = None, method="idct") -> None:
        super().__init__(N)
        self.resolution = (
            2 * N if resolution is None else resolution
        )  # number of points per cycle [0, 2pi)

        if method not in ["idct", "cos"]:
            raise ValueError("method must be one of 'idct', 'cos'")
        self.method = (
            self._basis_from_idct if method == "idct" else self._basis_from_cos
        )

        self.create_basis()

    def create_basis(self) -> None:
        self.basis = np.zeros((self.N, self.resolution))
        for i in range(self.N):
            self.basis[i, :] = self.method(i)

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian

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
        self.create_basis()

    def create_basis(self) -> None:
        Bx = CosineBasis(self.Nx)
        By = CosineBasis(self.Ny)
        self.basis = np.outer(Bx.basis, By.basis)

    def compute_jacobian(self, forward) -> None:
        pass


class PixelBasis(BaseBasis):
    def __init__(self, N: int) -> None:
        super().__init__(N)
        self.create_basis()

    def create_basis(self) -> None:
        self.basis = np.eye(self.N)

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian


class PixelBasis2D(BaseBasis):
    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self.create_basis()

    def create_basis(self) -> None:
        Bx = PixelBasis(self.Nx)
        By = PixelBasis(self.Ny)
        self.basis = np.outer(Bx.basis, By.basis)

    def compute_jacobian(self, forward) -> None:
        pass
