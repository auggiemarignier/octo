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
