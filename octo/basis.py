import numpy as np
from scipy.fft import idct
from typing import Callable


class BaseBasis:
    """
    Base class for basis functions.

    :param N: number of basis functions
    """

    def __init__(self, N: int) -> None:
        """
        N: number of basis functions
        """
        self.N = N
        self.basis = None  #: Matrix of basis functions. Each column is a basis function
        self.jacobian = None  #: Jacobian of measurement operator

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Expand coefficients X in terms of basis functions

        .. math::

            f(x) = \sum_{i=0}^{N-1} X_i \phi_i(x)

        :param X: vector of N coefficients
        """
        # self.basis is a matrix of shape (N, N) with each column representing a basis function
        return self.basis @ X

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Get the ith basis function

        :param i: index of basis function
        """
        return self.basis[:, i]

    def compute_jacobian(self, forward: Callable) -> None:
        """
        Compute the action of a forward measurement operator on the basis functions

        .. math::

            J_{ij} = \int G_i(\phi_j(x)) dx

        :param forward: measurement operator. Takes a vector of length N.
        """
        self.jacobian = np.vstack([forward(self.basis[:, i]) for i in range(self.N)]).T

    def _create_basis(self) -> None:
        """
        To be implemented for each new basis class.
        Should be called in __init__.

        Creates a matrix of basis functions of shape (N, N)
        self.basis[:, i] is the ith basis function
        i.e. each basis function is a column vector
        """
        pass


class CosineBasis(BaseBasis):
    """
    1D cosine basis functions

    :param method: method for creating basis functions. Either "idct" (default) or "cos".
        "idct" uses the :code:`scipy.fft.idct` function to create the basis functions.
        "cos" uses the formula

        .. math::

            \phi_i(x) = \\frac{1}{\sqrt{N/2}} \cos \left( \\frac{i \pi (2n+1)}{2N} \\right)

        where n = 0, 1, ..., N-1 and i = 0, 1, ..., N-1.
        Both produce the same orthonormal basis
    """

    def __init__(self, N: int, method: str = "idct") -> None:
        super().__init__(N)

        if method not in ["idct", "cos"]:
            raise ValueError("method must be one of 'idct', 'cos'")
        self.method = (
            self._basis_from_idct if method == "idct" else self._basis_from_cos
        )

        self._create_basis()

    def plot(self, figsize: tuple = (6, 4), show: bool = False):
        """
        Simple plotting routine.
        Basis functions are upsampled by a factor of 10 for plotting.

        :param figsize: figure size as per :matplotlib:
        :param show: show the figure
        """
        import matplotlib.pyplot as plt

        self._create_basis(_resolution=10 * self.N)
        plt.figure(figsize=figsize)
        for i, basis in enumerate(self.basis.T):
            plt.plot(i + basis)
        if show:
            plt.show()

        self._create_basis()  # reset basis

    def _create_basis(self, _resolution: int = None) -> None:
        """
        The _resolution argument is included for testing purposes. It is not
        intended to be used by the user. It represents the number of points in
        the range [0, pi) that the basis functions are evaluated at.
        """
        if _resolution is None:
            _resolution = self.N
        self.basis = np.zeros((_resolution, self.N))
        for i in range(self.N):
            self.basis[:, i] = self.method(i, _resolution)

    def _basis_from_idct(self, i, _resolution) -> np.ndarray:
        return idct(np.eye(_resolution)[i, :], norm="ortho")

    def _basis_from_cos(self, i, _resolution) -> np.ndarray:
        norm = np.sqrt(_resolution / 2)
        if i == 0:
            norm *= np.sqrt(2)
        n = 2 * np.arange(_resolution) + 1
        return np.cos(i * np.pi * n / (2 * _resolution)) / norm


class CosineBasis2D(BaseBasis):
    """
    2D cosine basis functions

    :param Nx: number of basis functions in x direction
    :param Ny: number of basis functions in y direction. If None, Ny = Nx.
    """

    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self._create_basis()

    def plot(self, figsize: tuple = (6, 4), show: bool = False):
        """
        Simple plotting routine.
        Basis functions are upsampled by a factor of 10 for plotting.
        Figure is divided into Nx x Ny subplots, each being one basis function, although this is not how they are stored in the basis matrix.

        :param figsize: figure size as per :matplotlib:
        :param show: show the figure
        """
        import matplotlib.pyplot as plt

        factor = 10  # upsampling factor

        Bx = CosineBasis(self.Nx)
        Bx._create_basis(_resolution=factor * self.Nx)
        By = CosineBasis(self.Ny)
        By._create_basis(_resolution=factor * self.Ny)

        basis_matrix = np.outer(Bx.basis.T, By.basis.T)
        plt.figure(figsize=figsize)
        plt.imshow(basis_matrix, cmap="RdBu")
        for x in range(1, self.Ny):
            plt.axvline(x * factor * self.Ny, color="k", ls="--")
        for y in range(1, self.Nx):
            plt.axhline(y * factor * self.Nx, color="k", ls="--")
        plt.axis(False)
        if show:
            plt.show()

    def _create_basis(self) -> None:
        Bx = CosineBasis(self.Nx)
        By = CosineBasis(self.Ny)
        self.basis = _unravel(np.outer(Bx.basis.T, By.basis.T), self.Nx, self.Ny)


class PixelBasis(BaseBasis):
    """
    1D pixel basis functions
    """

    def __init__(self, N: int) -> None:
        super().__init__(N)
        self._create_basis()

    def plot(self, figsize: tuple = (6, 4), show: bool = False):
        """
        Simple plotting routine.

        :param figsize: figure size as per :matplotlib:
        :param show: show the figure
        """
        import matplotlib.pyplot as plt

        x_fine = np.linspace(0, self.N, 1000)
        plt.figure(figsize=figsize)
        for i in range(self.N):
            basis_fine = np.zeros_like(x_fine)
            basis_fine[np.argmin(np.abs(x_fine - i))] = 0.95
            plt.plot(x_fine, i + basis_fine)

        if show:
            plt.show()

    def _create_basis(self) -> None:
        self.basis = np.eye(self.N)


class PixelBasis2D(BaseBasis):
    """
    2D pixel basis functions

    :param Nx: number of basis functions in x direction
    :param Ny: number of basis functions in y direction. If None, Ny = Nx.
    """

    def __init__(self, Nx: int, Ny: int = None) -> None:
        if Ny is None:
            Ny = Nx
        self.Nx = Nx
        self.Ny = Ny
        super().__init__(Nx * Ny)
        self._create_basis()

    def plot(self, figsize: tuple = (6, 4), show: bool = False):
        """
        Simple plotting routine.
        Figure is divided into Nx x Ny subplots, each being one basis function, although this is not how they are stored in the basis matrix.

        :param figsize: figure size as per :matplotlib:
        :param show: show the figure
        """
        import matplotlib.pyplot as plt

        Bx = PixelBasis(self.Nx)
        By = PixelBasis(self.Ny)
        basis_matrix = np.outer(Bx.basis, By.basis)

        plt.figure(figsize=figsize)
        plt.imshow(basis_matrix, cmap="binary")
        for x in range(1, self.Ny):
            plt.axvline(x * self.Ny, color="k", ls="--")
        for y in range(1, self.Nx):
            plt.axhline(y * self.Nx, color="k", ls="--")
        plt.axis(False)
        if show:
            plt.show()

    def _create_basis(self) -> None:
        Bx = PixelBasis(self.Nx)
        By = PixelBasis(self.Ny)
        self.basis = _unravel(np.outer(Bx.basis.T, By.basis.T), self.Nx, self.Ny)


def _unravel(basis_matrix: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Combining 2 1D basis classes by taking the outer product of their basis
    creates a matrix of 2D basis functions. This function unravels that matrix
    such that the 2D basis functions are flattened into a column vector.
    """
    unraveled = np.zeros((nx * ny, nx * ny))
    for i in range(nx):  # row
        for j in range(ny):  # column
            k = i * ny + j
            unraveled[:, k] = basis_matrix[
                i * nx : (i + 1) * nx, j * ny : (j + 1) * ny
            ].ravel()
    return unraveled


def _reravel(basis_matrix: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    The inverse of _unravel
    """
    reraveled = np.zeros((nx * nx, ny * ny))
    for i in range(nx):  # row
        for j in range(ny):  # column
            k = i * ny + j
            reraveled[i * nx : (i + 1) * nx, j * ny : (j + 1) * ny] = basis_matrix[
                :, k
            ].reshape((nx, ny))
    return reraveled
