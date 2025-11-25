# booz_xform_jax

`booz_xform_jax` is a pure Python implementation of the Boozer coordinate
transformation originally provided by the
[booz\_xform](https://github.com/hiddenSymmetries/booz_xform) package.  The
primary difference is that this version is written entirely in Python and
leverages the [JAX](https://github.com/google/jax) library for
high‑performance array operations, just‐in‐time (JIT) compilation, and
automatic differentiation.  Because the implementation avoids compiled C++
extensions and the Eigen linear algebra library, it can be installed via
`pip` on any platform that supports JAX.

The goal of this package is to provide a drop‑in replacement for the
`booz_xform` module used in optimization and analysis codes such as
[SIMSOPT](https://simsopt.readthedocs.io) while exposing the same inputs and
outputs.  The implementation follows the algorithm described in the
documentation of `booz_xform` and should reproduce the same results to
machine precision.  It has been designed so that users can freely compose
JAX transformations such as `jit`, `vmap`, and `grad` around the core
functions.

## Quick example

```python
from booz_xform_jax import BoozXform
import numpy as np

# Construct an instance.  At minimum you must specify the number of field
# periods (nfp) and whether the configuration is asymmetric.
bx = BoozXform(nfp=5, asym=False)

# Supply Fourier mode metadata.  The xm and xn arrays describe the m and n
# indices of the non‑Nyquist modes, and xm_nyq/xn_nyq describe the Nyquist
# modes.  These come from VMEC output (see the booz_xform documentation
# for details).
bx.mpol = 32
bx.ntor = 32
bx.mnmax = len(xm)
bx.xm = xm
bx.xn = xn
bx.mpol_nyq = int(xm_nyq[-1])
bx.ntor_nyq = int(xn_nyq[-1] // nfp)
bx.mnmax_nyq = len(xm_nyq)
bx.xm_nyq = xm_nyq
bx.xn_nyq = xn_nyq

# Initialise the transformation with data from VMEC.  The arrays
# rmnc, rmns, etc. should be shaped `(mnmax, ns)` or `(mnmax_nyq, ns)` as
# appropriate.  iotas has length ns and contains the rotational transform
# at each radial surface.  The first radial index (s=0) is omitted by
# convention since it contains only zeros.
bx.init_from_vmec(iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns,
                  bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns)

# Run the transformation on all registered surfaces (by default all half
# radial surfaces).  This computes the Boozer Fourier spectrum on the
# surfaces specified by `compute_surfs`.
bx.run()

# The results are available as attributes on the object.  For example,
# `bmnc_b` contains the cos(m*θ − n*ζ) harmonics of the magnetic field strength
# expressed in Boozer coordinates.
print(bx.bmnc_b.shape)  # (mnboz, number of computed surfaces)
```

This example glosses over many details—see the docstrings in
`booz_xform_jax/booz_xform.py` for a full description of the inputs and
outputs.  The API has been kept intentionally close to that of the
original C++ implementation so that existing code can be ported with
minimal changes.

## Features

The JAX implementation supports the full range of functionality of the
original ``booz_xform`` program:

* **Reading VMEC equilibria** via the :py:meth:`read_wout`
  method.  This method reads a VMEC ``wout`` file and populates all
  Fourier metadata and spectral arrays.  You can also supply arrays
  directly using :py:meth:`init_from_vmec`.

* **Selecting radial surfaces** using either the ``compute_surfs``
  attribute or the convenience method :py:meth:`register_surfaces`.
  Surfaces may be specified by integer half–grid indices or by their
  normalized toroidal flux values.

* **Computing Boozer spectra** with :py:meth:`run`.  The transformation
  can be JIT–compiled or vectorised using JAX's transformations.  The
  resulting Fourier coefficients (``bmnc_b``, ``rmnc_b``, ``zmns_b``, etc.)
  exactly mirror those produced by the C++ code and are stored in
  NumPy arrays for interoperability.

* **Writing and reading boozmn files** via
  :py:meth:`write_boozmn` and :py:meth:`read_boozmn`.  The NetCDF
  format is identical to that used by the Fortran/C++ implementation so
  that downstream tools (including ``simsopt`` and ``booz_xform`` itself)
  can read the output.

* **Plotting utilities** in :mod:`booz_xform_jax.plots`.  Four functions
  (``surfplot``, ``symplot``, ``modeplot``, ``wireplot``) are provided to
  visualise the magnetic field strength and coordinate curves.  These
  functions accept either a :class:`BoozXform` instance or the path to
  a previously saved boozmn file.  Matplotlib is required to use the
  plotting functions.

* **Unit tests** drawn from the original repository have been ported to
  this package and can be run with ``pytest``.  They exercise the
  transformation, I/O and plotting logic and compare results to known
  reference files.

## Installation

To install the package from a local checkout, run::

    pip install .

JAX will be pulled in automatically along with its runtime.  If you
intend to run the plotting utilities you should ensure that
``matplotlib`` and ``plotly`` are available; they are declared as
dependencies in the package metadata.

## License

This project is licensed under the terms of the MIT license.  See the
included `LICENSE` file for details.