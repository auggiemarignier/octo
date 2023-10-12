from octo.basis import BaseBasis, CosineBasis, PixelBasis
import numpy as np
from scipy.optimize import minimize
from typing import List, Callable


class OvercompleteBasis:
    def __init__(
        self,
        data: np.ndarray,
        bases: List[BaseBasis],
        covariance: np.ndarray = None,
        bweights: List[float] = None,
        rweight: float = None,
    ) -> None:
        """
        data: observed data to be fitted
        bases: list of basis objects
        covariance: covariance matrix for data.  Default is identity matrix.
        bweights: list of weights for each basis.  Default is 1.0 for each basis.
        rweight: Regularisation weight. Default is 1.0.
        """
        self.data = data
        self.bases = bases
        self.bweights = bweights if bweights is not None else [1.0 for _ in bases]
        self.rweight = rweight if rweight is not None else 1.0
        self.covariance = covariance if covariance is not None else np.eye(len(data))

        self._check_jacobians()

    def cost(self, x: np.ndarray) -> float:
        """
        x: proposed solution to be compared with observed data
        """
        misfit = np.zeros_like(self.data)
        for basis, _x in zip(self.bases, self._split(x)):
            misfit += basis.jacobian @ _x
        misfit -= self.data
        weighted_squared_misfit = misfit.T @ np.linalg.inv(self.covariance) @ misfit
        cost = weighted_squared_misfit + self.rweight * np.linalg.norm(x, 1)
        return cost

    def update_jacobians(self, forward: Callable):
        for basis in self.bases:
            basis.compute_jacobian(forward)

    def _check_jacobians(self):
        for basis in self.bases:
            assert basis.jacobian is not None, "Jacobian not computed for basis"

    def _split(self, x):
        """
        x is a vector of coefficients for the overcomplete basis.
        Split x into vectors of coefficients for each basis.
        """
        split = []
        for basis in self.bases:
            split.append(x[: basis.N])
            x = x[basis.N :]
        return np.array(split)


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
