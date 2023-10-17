.. octo documentation master file, created by
   sphinx-quickstart on Tue Oct 17 10:11:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OverComplete TOmography (OCTO)
================================

Why limit yourself to a single basis when you can have them all? 
OCTO is a package for combining multiple bases to perform tomographic imaging.
By combining mutliple families of basis functions and a sparsity-promoting prior, OCTO can recover features that would be missed by individual bases alone.
For example, we can recover combinations of localized and smooth features.

The approach implemented here was first described in `Turunctur et al., (2023) <https://academic.oup.com/rasti/article/2/1/207/7146838>`_.
Pixel and Cosine basis functions are implemented in 1D and 2D, and the package is designed to be easily extended to other bases.
A main class, :class:`octo.Octo`, is used to combine the bases together and perform the tomographic reconstruction.

Installation
============

.. code-block:: bash

    $ pip install .


Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Modules

   modulesrst/basis.rst 
   modulesrst/overcomplete.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/example

.. toctree::
   :maxdepth: 1
   :caption: About



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
