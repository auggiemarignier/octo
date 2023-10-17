Basis
=====

This module provides a simple way to create a basis for a vector space.
The :class:`BaseBasis` class is the base class for all bases.
It provides a way to apply a set of coefficeints to a basis, access to individual basis functions, and a way to compute a Jacobian matrix combining the basis functions with a forward measurement model.


.. autoclass:: basis.BaseBasis
    :members:
    :special-members: __call__, __getitem__
    :private-members: _create_basis


.. autoclass:: basis.CosineBasis
    :members:

.. autoclass:: basis.PixelBasis
    :members:

.. autoclass:: basis.CosineBasis2D
    :members:

.. autoclass:: basis.PixelBasis2D
    :members: