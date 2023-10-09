from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.fft import dct


class BaseBasis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, N: int) -> None:
        """
        N: number of basis functions
        """
        self.N = N
        self.jacobian = None
        self.create_basis()

    @abstractmethod
    def create_basis(self) -> None:
        pass

    @abstractmethod
    def compute_jacobian(self, forward) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self, N: int) -> None:
        super().__init__(N)
        self.resolution = N  # number of points to evaluate the basis at

    def create_basis(self) -> None:
        self.basis = np.zeros((self.N, self.resolution))
        for i in range(self.N):
            self.basis[i, :] = dct(np.eye(self.resolution)[i], type=2, norm="ortho")

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian


class PixelBasis(BaseBasis):
    def __init__(self, N: int) -> None:
        super().__init__(N)

    def create_basis(self) -> None:
        self.basis = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.basis[i, j] = 1 if i == j else 0

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian
