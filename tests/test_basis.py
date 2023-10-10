import numpy as np
import pytest
from octo.basis import CosineBasis, PixelBasis, CosineBasis2D, PixelBasis2D


def _is_othornomal_basis(basis_matrix):
    N = basis_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            dot_product = np.dot(basis_matrix[i, :], basis_matrix[j, :])
            expected = 1.0 if i == j else 0.0
            assert np.isclose(dot_product, expected), f"{i,j}"


def test_cosine_basis_implementations():
    N = 20
    res = 10000
    b1 = CosineBasis(N, res, method="idct")
    b2 = CosineBasis(N, res, method="cos")
    assert np.allclose(b1.basis, b2.basis)


@pytest.mark.parametrize("method", ["idct", "cos"])
def test_cosine_basis_orthonormal(method):
    N = 10
    cosine_basis = CosineBasis(N, method=method)
    basis_matrix = cosine_basis.basis
    _is_othornomal_basis(basis_matrix)


def test_pixel_basis_orthonormal():
    N = 10
    pixel_basis = PixelBasis(N)
    basis_matrix = pixel_basis.basis
    _is_othornomal_basis(basis_matrix)


def test_cosine2d_basis_orthonormal():
    Nx = 10
    Ny = 10
    cosine2d_basis = CosineBasis2D(Nx, Ny)
    basis_matrix = cosine2d_basis.basis
    _is_othornomal_basis(basis_matrix)


def test_pixel2d_basis_orthonormal():
    Nx = 10
    Ny = 10
    pixel2d_basis = PixelBasis2D(Nx, Ny)
    basis_matrix = pixel2d_basis.basis
    _is_othornomal_basis(basis_matrix)
