from abc import ABCMeta, abstractmethod
import numpy as np


class BaseBasis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, N: int, M: int) -> None:
        """
        N: number of unknowns
        M: number of measurements
        """
        self.N = N
        self.M = M
        self.jacobian = None
        self.create_basis()

    @abstractmethod
    def create_basis(self) -> None:
        pass

    @abstractmethod
    def compute_jacobian(self, forward) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self, N: int, M: int) -> None:
        super().__init__(N, M)

    def create_basis(self) -> None:
        self.basis = np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                self.basis[i, j] = np.cos(2 * np.pi * i * j / self.N)

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian


class PixelBasis(BaseBasis):
    def __init__(self, N: int, M: int) -> None:
        super().__init__(N, M)

    def create_basis(self) -> None:
        self.basis = np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                self.basis[i, j] = 1 if i == j else 0

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian
