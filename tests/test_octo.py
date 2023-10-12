from octo.octo import OvercompleteBasis
from octo.basis import CosineBasis, PixelBasis
import numpy as np
from numpy.random import default_rng
import pytest

rng = default_rng(42)


def forward(x: np.ndarray) -> np.ndarray:
    return x


@pytest.fixture
def N():
    return 10


@pytest.fixture
def bases(N):
    _bases = [CosineBasis(N), PixelBasis(N)]
    for b in _bases:
        b.compute_jacobian(forward)
    return _bases


@pytest.fixture
def mc(bases):
    """
    True random cosine coefficients
    """
    return rng.random(bases[0].N)


@pytest.fixture
def mp(bases):
    """
    True random pixel coefficients
    """
    return rng.random(bases[1].N)


@pytest.fixture
def data(bases, mc, mp):
    field = bases[0](mc) + bases[1](mp)
    return forward(field)


def test_overcomplete_init(data, N):
    bases = [CosineBasis(N), PixelBasis(N)]
    bweights = rng.random(2)
    rweight = rng.random(1)
    with pytest.raises(AssertionError):
        overcomplete_basis = OvercompleteBasis(data, bases, bweights, rweight)

    for b in bases:
        b.compute_jacobian(forward)
    overcomplete_basis = OvercompleteBasis(data, bases, bweights, rweight)

    assert np.allclose(overcomplete_basis.data, data)
    assert np.array_equal(overcomplete_basis.bases, bases)
    assert np.array_equal(overcomplete_basis.bweights, bweights)
    assert np.array_equal(overcomplete_basis.rweight, rweight)


def test_overcomplete_cost(data, bases, mc, mp):
    overcomplete_basis = OvercompleteBasis(data, bases, rweight=0.0)
    cost = overcomplete_basis.cost(np.concatenate([mc, mp]))
    assert cost == pytest.approx(0.0)
