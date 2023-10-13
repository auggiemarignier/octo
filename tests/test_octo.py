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
        overcomplete_basis = OvercompleteBasis(
            data, bases, bweights=bweights, rweight=rweight
        )

    for b in bases:
        b.compute_jacobian(forward)
    overcomplete_basis = OvercompleteBasis(
        data, bases, bweights=bweights, rweight=rweight
    )

    assert np.allclose(overcomplete_basis.data, data)
    assert np.array_equal(overcomplete_basis.bases, bases)
    assert np.array_equal(overcomplete_basis.bweights, bweights / np.sum(bweights))
    assert np.array_equal(overcomplete_basis.rweight, rweight)


def test_overcomplete_cost(data, bases, mc, mp):
    overcomplete_basis = OvercompleteBasis(data, bases, rweight=0.0)
    cost = overcomplete_basis.cost(np.concatenate([mc, mp]))
    assert cost == pytest.approx(0.0)


def test_overcomplete_combined_jacobian(data, bases, mc, mp):
    overcomplete_basis = OvercompleteBasis(data, bases, rweight=0.0)
    assert overcomplete_basis.jacobian.shape == (data.size, mc.size + mp.size)


def test_data_misfit_gradient(data, bases, mc, mp):
    # Generate a random input vector
    x = np.random.rand(mc.size + mp.size)

    # Compute the misfit gradient
    overcomplete_basis = OvercompleteBasis(data, bases, rweight=0.0)
    gradient = overcomplete_basis.data_misfit_gradient(x)

    # Check that the shape of the misfit gradient is correct
    assert gradient.shape == (mc.size + mp.size,)

    # Calculate the expected misfit gradient basis by basis
    split = overcomplete_basis._split(x)
    expected_misfit = (
        np.sum([b.jacobian @ _x for b, _x in zip(bases, split)], axis=0) - data
    )
    expected_gradient = np.hstack([b.jacobian.T @ expected_misfit for b in bases])
    assert np.allclose(gradient, expected_gradient)


def test_l1_gradient(data, bases, mc, mp):
    """
    With Identity covariance operator, the weighting norm is just the norm of the basis jacboian.
    The l1 gradient is the signs of the coefficients (ignoring zeros).
    """
    # Generate a random input vector
    x = np.random.randn(mc.size + mp.size)

    # Compute the l1 gradient
    overcomplete_basis = OvercompleteBasis(data, bases, rweight=2.0)
    gradient = overcomplete_basis.l1_reg_gradient(x)

    # Check that the shape of the l1 gradient is correct
    assert gradient.shape == (mc.size + mp.size,)

    # Calculate the expected l1 gradient basis by basis
    l1_grads = []
    for b, bw, _x in zip(
        overcomplete_basis.bases,
        overcomplete_basis.bweights,
        overcomplete_basis._split(x),
    ):
        norm = np.linalg.norm(
            np.sqrt(np.linalg.inv(overcomplete_basis.covariance)) @ b.jacobian, 2
        )
        l1_grad = overcomplete_basis.rweight * bw * norm * np.sign(_x)
        l1_grads.append(l1_grad)
    expected_gradient = np.hstack(l1_grads)

    assert np.allclose(gradient, expected_gradient)
