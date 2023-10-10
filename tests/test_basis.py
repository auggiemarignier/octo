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


def test_cosine_basis_implementations(N):
    res = 10000
    b1 = CosineBasis(N, res, method="idct")
    b2 = CosineBasis(N, res, method="cos")
    assert np.allclose(b1.basis, b2.basis)


@pytest.mark.parametrize("basis", [PixelBasis, CosineBasis])
def test_1D_basis_orthonormal(basis, N):
    _basis = basis(N)
    _is_othornomal_basis(_basis.basis)


@pytest.mark.parametrize("basis", [PixelBasis2D, CosineBasis2D])
def test_2d_basis_orthonormal(basis, Nx, Ny):
    _basis = basis(Nx, Ny)
    _is_othornomal_basis(_basis.basis)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _N = 10
    cosine_basis = CosineBasis(_N, 100)
    basis_matrix = cosine_basis.basis
    plt.plot(basis_matrix.T)
    plt.show()

    pixel_basis = PixelBasis(_N)
    basis_matrix = pixel_basis.basis
    plt.plot(basis_matrix.T)
    plt.show()

    _Nx = 10
    _Ny = 5
    cosine_basis = CosineBasis2D(_Nx, _Ny)
    basis_matrix = cosine_basis.basis
    plt.imshow(basis_matrix)
    plt.show()

    pixel_basis = PixelBasis2D(_Nx, _Ny)
    basis_matrix = pixel_basis.basis
    plt.imshow(basis_matrix)
    plt.show()
