from abc import ABCMeta, abstractmethod


class BaseBasis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_basis(self, x: float) -> float:
        pass

    @abstractmethod
    def compute_jacobian(self, x: float) -> float:
        pass


class CosineBasis(BaseBasis):
    def __init__(self) -> None:
        pass

    def create_basis(self, x: float) -> float:
        pass

    def compute_jacobian(self, x: float) -> float:
        pass


class PixelBasis(BaseBasis):
    def __init__(self) -> None:
        pass

    def create_basis(self, x: float) -> float:
        pass

    def compute_jacobian(self, x: float) -> float:
        pass
