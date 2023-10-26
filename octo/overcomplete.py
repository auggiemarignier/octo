from octo.basis import BaseBasis
import numpy as np
from typing import List


class OvercompleteBasis:
    """
    Overcomplete basis class.
    Combines bases and provides functions for optimisation

    :param data: observed data to be fitted
    :param bases: list of basis objects
    :param regularisation: type of regularisation.  Default is l1. Options are ['l1', 'l2']
    :param covariance: covariance matrix for data :math:`C_d`.  Default is identity matrix.
    :param bweights: list of weights :math:`\\beta_k` for each basis.  Default is even weight for each basis. Sum to 1 is enforced.
    :param rweight: Regularisation weight :math:`\mu`. Default is 1.0.
    """

    def __init__(
        self,
        data: np.ndarray,
        bases: List[BaseBasis],
        regularisation: str = "l1",
        covariance: np.ndarray = None,
        bweights: List[float] = None,
        rweight: float = None,
    ) -> None:
        """ """
        self.data = data
        self.bases = bases
        self._check_kernels()
        self._combine_kernels()

        self.bweights = bweights if bweights is not None else [1.0 for _ in bases]
        self._check_bweights()

        self.rweight = rweight if rweight is not None else 1.0
        self.covariance = covariance if covariance is not None else np.eye(len(data))
        self.invcov = np.linalg.inv(self.covariance)

        if regularisation == "l1":
            self.reg = self.l1_reg
            self.reg_gradient = self.l1_reg_gradient
            self._precompute_l1_wieghting_norms()
            self._precompute_cost_hessian()
        elif regularisation == "l2":
            raise NotImplementedError("L2 regularisation not yet implemented")
        else:
            raise ValueError(f"Unknown regularisation {regularisation}")

    def cost(self, x: np.ndarray) -> float:
        """
        Overall objective function to be optimised.
        A combination of data misfit and regularisation.

        .. math::

            \Theta(x) = \chi(x) + \mu \ell_1(x)

        :param x: proposed solution to be compared with observed data
        """
        return self.data_misfit(x) + self.reg(x)

    def cost_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of overall objective function to be optimised.

        :param x: proposed solution to be compared with observed data
        """
        return self.data_misfit_gradient(x) + self.reg_gradient(x)

    def cost_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Hessian of overall objective function to be optimised.

        :param x: proposed solution to be compared with observed data
        """
        return self._cost_hessian

    def data_misfit(self, x: np.ndarray) -> float:
        """
        Squared data misfit between observed data and proposed solution, weighted by the inverse covariance matrix.

        .. math::

            \chi(x) = \\frac{1}{2} (d - Gx)^T C_d^{-1} (d - Gx)

        :param x: proposed solution to be compared with observed data
        """
        misfit = self.data - self.kernel @ x
        return misfit.T @ self.invcov @ misfit / 2.0

    def data_misfit_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of squared data misfit between observed data and proposed solution.

        :param x: proposed solution to be compared with observed data
        """
        return self.kernel.T @ self.invcov @ (self.kernel @ x - self.data)

    def l1_reg(self, x: np.ndarray) -> float:
        """
        L1 norm regularisation with a norm weight to account for different basis units

        .. math::

            \ell_1(x) = \sum_{k=1}^K \\beta_k \left\| C_d^{-1/2}G^k \\right\|_2\left\| x^k \\right\|_1

        :param x: proposed solution
        """
        l1 = np.linalg.norm(self._split(x), 1, axis=1)[:, np.newaxis]
        return self.rweight * np.sum(self.bweights * self.l1_weighting_norms * l1)

    def l1_reg_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of L1 norm regularisation.
        """
        split = self._split(x)
        l1_grads = np.sign(split)
        gradient = self.bweights * self.l1_weighting_norms * l1_grads
        return self.rweight * np.hstack(gradient)

    def _combine_kernels(self):
        self.kernel = np.hstack([b.kernel for b in self.bases])

    def _check_kernels(self):
        for basis in self.bases:
            assert basis.kernel is not None, "kernel not computed for basis"

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
                np.sqrt(self.invcov) @ basis.kernel, 2
            )

    def _precompute_cost_hessian(self):
        # Hessian is constant so can be precomputed
        # L2 datamisfit hessian = G^TC^{-1}G
        # L1 regularisation hessian = 0
        self._cost_hessian = self.kernel.T @ self.invcov @ self.kernel
