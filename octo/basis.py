from abc import ABCMeta, abstractmethod
import numpy as np


class BaseBasis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_basis(self) -> None:
        pass

    @abstractmethod
    def compute_jacobian(self, forward) -> None:
        pass


class CosineBasis(BaseBasis):
    def __init__(self) -> None:
        self.jacobian = None

    def create_basis(self) -> None:
        pass

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian


class PixelBasis(BaseBasis):
    def __init__(self) -> None:
        self.jacobian = None

    def create_basis(self) -> None:
        pass

    def compute_jacobian(self, forward) -> None:
        self.jacobian = np.array([[forward(0.0)], [forward(1.0)]])
        return self.jacobian
