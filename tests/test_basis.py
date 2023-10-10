import numpy as np
import pytest
from octo.basis import CosineBasis, PixelBasis


@pytest.mark.parametrize(
    "basis_class, N, M, expected_basis",
    [
        (CosineBasis, 3, 2, np.array([[1.0, 0.5, 0.0], [1.0, 0.0, -1.0]])),
        (
            PixelBasis,
            4,
            3,
            np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
        ),
    ],
)
def test_basis_attribute(basis_class, N, M, expected_basis):
    basis = basis_class(N, M)
    assert np.allclose(basis.basis, expected_basis)


def test_plot_cosine_basis_functions():
    import matplotlib.pyplot as plt

    N = 20
    res = 10000
    cosine_basis = CosineBasis(N, res)
    basis_matrix = cosine_basis.basis
    b2 = np.zeros_like(basis_matrix)
    for i in range(N):
        b2[i, :] = np.cos(i * np.linspace(0, np.pi, res)) / np.sqrt(res / 2)
        if i == 0:
            b2[i, :] /= np.sqrt(2)
        plt.plot(i + basis_matrix[i, :])
        plt.plot(i + b2[i, :], ls="--")
    plt.show()
    assert np.allclose(basis_matrix, b2, rtol=1e-3, atol=1e-8)


def test_cosine_basis_orthonormal():
    N = 10
    cosine_basis = CosineBasis(N)
    basis_matrix = cosine_basis.basis
    for i in range(N):
        for j in range(N):
            dot_product = np.dot(basis_matrix[i, :], basis_matrix[j, :])
            if i == j:
                assert np.isclose(dot_product, 1.0), f"{i,j}"
            else:
                assert np.isclose(dot_product, 0.0), f"{i,j}"
