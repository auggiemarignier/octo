from octo.basis import BaseBasis, CosineBasis, PixelBasis
import numpy as np
from scipy.optimize import minimize
from typing import List, Callable


class OvercompleteBasis:
    def __init__(
        self,
        data: np.ndarray,
        bases: List[BaseBasis],
        regularisation: str = "l1",
        covariance: np.ndarray = None,
        bweights: List[float] = None,
        rweight: float = None,
    ) -> None:
        """
        data: observed data to be fitted
        bases: list of basis objects
        regularisation: type of regularisation.  Default is l1. Options are ['l1', 'l2']
        covariance: covariance matrix for data.  Default is identity matrix.
        bweights: list of weights for each basis.  Default is even weight for each basis. Sum to 1 is enforced.
        rweight: Regularisation weight. Default is 1.0.
        """
        self.data = data
        self.bases = bases
        self.bweights = bweights if bweights is not None else [1.0 for _ in bases]
        self.rweight = rweight if rweight is not None else 1.0
        self.covariance = covariance if covariance is not None else np.eye(len(data))

        if regularisation == "l1":
            self.reg = self.l1_reg
        elif regularisation == "l2":
            raise NotImplementedError("L2 regularisation not yet implemented")
        else:
            raise ValueError(f"Unknown regularisation {regularisation}")

        self._check_jacobians()
        self._combine_jacobians()
        self._check_bweights()

    def cost(self, x: np.ndarray) -> float:
        """
        x: proposed solution to be compared with observed data
        """
        return self.data_misfit(x) + self.rweight * self.reg(x)

    def data_misfit(self, x: np.ndarray) -> float:
        """
        x: proposed solution to be compared with observed data
        """
        misfit = self.data - self.jacobian @ x
        return misfit.T @ np.linalg.inv(self.covariance) @ misfit

    def l1_reg(self, x: np.ndarray) -> float:
        """
        L1 norm regularisation with a norm weight to account for different basis units
        """
        l1 = 0
        for b, bw, _x in zip(self.bases, self.bweights, self._split(x)):
            norm = np.linalg.norm(
                np.sqrt(np.linalg.inv(self.covariance)) @ b.jacobian, 2
            )
            l1 += bw * norm * np.linalg.norm(_x, 1)
        return l1

    def _combine_jacobians(self):
        self.jacobian = np.hstack([b.jacobian for b in self.bases])

    def _check_jacobians(self):
        for basis in self.bases:
            assert basis.jacobian is not None, "Jacobian not computed for basis"

    def _check_bweights(self):
        if len(self.bweights) != len(self.bases):
            raise ValueError("Number of basis weights does not match number of bases")
        if np.sum(self.bweights) != 1.0:
            self.bweights /= np.sum(self.bweights)

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
