# OverComplete TOmography

This repo contains an implementation of overcomplete tomography as detailed in [Turunctur et al., 2023](https://academic.oup.com/rasti/article/2/1/207/7146838).
Overcomplete tomography combines multiple families of basis functions, ideally each with different properties which when combined allows one to resolve a wide range of image features and characteristics.

## Installation

From source

```bash
poetry install
```

or

```bash
pip install .
```

## Usage

A complete synthetic example is shown in `examples/example.ipynb`.

The main idea is to create individual basis objects and combine them into one.
`octo.basis` contains implementations of 1D and 2D pixel and cosine basis functions.
Once the bases have been constructed, compute the individual kernel matrices given the forward measurement operator, i.e. the mapping from image to data space.
An example path integral operator is implemented in the example notebook.

`octo.octo` contains the `OvercompleteBasis` class, which is the main workhorse of the package.
It combines the individual bases to create an overall kernel matrix, calculates data misfits and sparsity regularisation and their derivatives.
`OvercompleteBasis.cost` is the function to be optimised, for example using `scipy.optimize.minimize`.
`OvercompleteBasis.cost_gradient` may also be helpful for this.