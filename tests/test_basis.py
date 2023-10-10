import numpy as np
import pytest
from octo.basis import CosineBasis, PixelBasis


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
    for i in range(N):
        for j in range(N):
            dot_product = np.dot(basis_matrix[i, :], basis_matrix[j, :])
            if i == j:
                assert np.isclose(dot_product, 1.0), f"{i,j}"
            else:
                assert np.isclose(dot_product, 0.0), f"{i,j}"
