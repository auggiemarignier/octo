import numpy as np
import pytest
from octo.basis import CosineBasis, PixelBasis, CosineBasis2D, PixelBasis2D


@pytest.fixture
def N():
    return 10


@pytest.fixture
def Nx():
    return 10


@pytest.fixture
def Ny():
    return 5


def _is_othornomal_basis(basis_matrix):
    N = basis_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            dot_product = np.dot(basis_matrix[i, :], basis_matrix[j, :])
            expected = 1.0 if i == j else 0.0
            assert np.isclose(dot_product, expected), f"{i,j}"


def test_ravel_unravel(Nx, Ny):
    from octo.basis import _unravel, _reravel
    from numpy.random import default_rng

    rng = default_rng(0)

    # The matrix that gets unravelled is the outer product of 2 1D basis
    # each 1D basis has N^2 elements, so the outer product has shape Nx^2 x Ny^2
    basis_matrix = rng.random((Nx * Nx, Ny * Ny))
    unraveled = _unravel(basis_matrix, Nx, Ny)
    reraveled = _reravel(unraveled, Nx, Ny)
    assert np.allclose(basis_matrix, reraveled)


def test_cosine_basis_implementations(N):
    b1 = CosineBasis(N, method="idct")
    b2 = CosineBasis(N, method="cos")
    assert np.allclose(b1.basis, b2.basis)


@pytest.mark.parametrize("basis", [PixelBasis, CosineBasis])
def test_1D_basis_orthonormal(basis, N):
    _basis = basis(N)
    _is_othornomal_basis(_basis.basis)


@pytest.mark.parametrize("basis", [PixelBasis2D, CosineBasis2D])
def test_2D_basis_orthonormal(basis, Nx, Ny):
    _basis = basis(Nx, Ny)
    _is_othornomal_basis(_basis.basis)


@pytest.mark.parametrize("basis", [PixelBasis, CosineBasis])
def test_1D_basis_jacboian(basis, N):
    ndata = 12

    def _forward(X):
        return np.eye(ndata, N).dot(X)

    _basis = basis(N)
    _basis.compute_jacobian(_forward)

    assert _basis.jacobian.shape == (ndata, N)


@pytest.mark.parametrize("basis", [PixelBasis2D, CosineBasis2D])
def test_2D_basis_jacboian(basis, Nx, Ny):
    ndata = 12

    def _forward(X):
        return np.eye(ndata, Nx * Ny).dot(X)

    _basis = basis(Nx, Ny)
    _basis.compute_jacobian(_forward)

    assert _basis.jacobian.shape == (ndata, Nx * Ny)


if __name__ == "__main__":
    _N = 10
    cosine_basis = CosineBasis(_N)
    cosine_basis.plot()

    pixel_basis = PixelBasis(_N)
    pixel_basis.plot()

    _Nx = 5
    _Ny = 3
    cosine_basis = CosineBasis2D(_Nx, _Ny)
    cosine_basis.plot()

    pixel_basis = PixelBasis2D(_Nx, _Ny)
    pixel_basis.plot()
