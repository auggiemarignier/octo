from octo.basis import BaseBasis, CosineBasis, PixelBasis
import numpy as np
from scipy.optimize import minimize
from typing import List


class OvercompleteBasis:
    def __init__(self, bases: List[BaseBasis]) -> None:
        self.bases = bases

    def cost(self, x: float, y: float) -> float:
        basis_val = self.basis.create_basis()
        jacobian = self.basis.compute_jacobian(forward=lambda x: x**2)
        return np.sum((y - basis_val) ** 2) + np.sum(jacobian)


def minimize_cost(overcomplete_basis: OvercompleteBasis, x0: float, y: float) -> float:
    res = minimize(overcomplete_basis.cost, x0=x0, args=(y,))
    return res.fun


def main():
    cosine_basis = CosineBasis()
    pixel_basis = PixelBasis()
    overcomplete_basis = OvercompleteBasis([cosine_basis, pixel_basis])
    x0 = 0.5
    y = 1.0
    min_cost = minimize_cost(overcomplete_basis, x0, y)
    print(f"Minimum cost: {min_cost}")


if __name__ == "__main__":
    main()
