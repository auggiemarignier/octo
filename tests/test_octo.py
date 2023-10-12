from octo.octo import OvercompleteBasis
from octo.basis import CosineBasis, PixelBasis
import numpy as np
from numpy.random import default_rng
import pytest

rng = default_rng(42)


def test_overcomplete_init():
    N = 10
    data = rng.random(N)
    bases = [CosineBasis(N), PixelBasis(N)]
    bweights = [1.0, 1.0]
    rweight = 1.0
    with pytest.raises(AssertionError):
        overcomplete_basis = OvercompleteBasis(data, bases, bweights, rweight)

    for b in bases:
        b.compute_jacobian(lambda x: np.eye(N).dot(x))
    overcomplete_basis = OvercompleteBasis(data, bases, bweights, rweight)

    assert np.allclose(overcomplete_basis.data, data)
    assert overcomplete_basis.bases == bases
    assert overcomplete_basis.bweights == bweights
    assert overcomplete_basis.rweight == rweight
