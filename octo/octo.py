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
        self._check_jacobians()
        self._combine_jacobians()

        self.bweights = bweights if bweights is not None else [1.0 for _ in bases]
        self._check_bweights()

        self.rweight = rweight if rweight is not None else 1.0
        self.covariance = covariance if covariance is not None else np.eye(len(data))
        self.invcov = np.linalg.inv(self.covariance)

        if regularisation == "l1":
            self.reg = self.l1_reg
            self.reg_gradient = self.l1_reg_gradient
            self._precompute_l1_wieghting_norms()
        elif regularisation == "l2":
            raise NotImplementedError("L2 regularisation not yet implemented")
        else:
            raise ValueError(f"Unknown regularisation {regularisation}")

    def cost(self, x: np.ndarray) -> float:
        """
        x: proposed solution to be compared with observed data
        """
        return self.data_misfit(x) + self.reg(x)

    def data_misfit(self, x: np.ndarray) -> float:
        """
        x: proposed solution to be compared with observed data
        """
        misfit = self.data - self.jacobian @ x
        return misfit.T @ self.invcov @ misfit / 2.0

    def data_misfit_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        x: proposed solution to be compared with observed data
        """
        return self.jacobian.T @ self.invcov @ (self.jacobian @ x - self.data)

    def l1_reg(self, x: np.ndarray) -> float:
        """
        L1 norm regularisation with a norm weight to account for different basis units
        """
        l1 = np.linalg.norm(self._split(x), 1, axis=1)[:, np.newaxis]
        return self.rweight * np.sum(self.bweights * self.l1_weighting_norms * l1)

    def l1_reg_gradient(self, x: np.ndarray) -> np.ndarray:
        split = self._split(x)
        l1_grads = np.sign(split)
        gradient = self.bweights * self.l1_weighting_norms * l1_grads
        return self.rweight * np.hstack(gradient)

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
        self.bweights = self.bweights[:, np.newaxis]

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

    def _precompute_l1_wieghting_norms(self):
        self.l1_weighting_norms = np.zeros((len(self.bases), 1))
        for i, basis in enumerate(self.bases):
            self.l1_weighting_norms[i] = np.linalg.norm(
                np.sqrt(self.invcov) @ basis.jacobian, 2
            )
