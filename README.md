# booz\_xform\_jax

A high-performance, JAX-based reimplementation of the classic **BOOZ\_XFORM** code for transforming VMEC equilibria into **Boozer coordinates** and computing Boozer-space Fourier spectra of the magnetic field strength \|B\| and related geometric quantities.

> **Status:** Research-grade, under active development. Interfaces and internals may change.
>
> This repository aims to be:
> - a **drop-in, differentiable analogue** of the C++/Fortran BOOZ\_XFORM used by `simsopt` and related tools;
> - a **pedagogical reference implementation** showing how the Boozer transform can be expressed in modern array-based, differentiable form.

---

## 1. Background and motivation

Design and optimisation of stellarators heavily exploit magnetic coordinates adapted to the field lines. Among them, **Boozer coordinates** are particularly important: in Boozer coordinates, the magnetic field takes the form
\[
\mathbf{B} = \nabla \psi \times \nabla \theta_B + \iota(\psi) \nabla \zeta_B \times \nabla \psi,
\]
with simple, nearly straight field lines and a convenient representation of quasisymmetry, quasi-isodynamicity, and omnigenity.

Most optimisation pipelines start from an MHD equilibrium computed by **VMEC** (a spectral code for 3D MHD equilibria). VMEC outputs the equilibrium in a VMEC-specific poloidal/toroidal angle system and in Fourier space. Post-processing codes like **BOOZ\_XFORM** then convert this VMEC representation into Boozer coordinates and compute Fourier spectra of \|B\| and other geometric quantities on flux surfaces.

### 1.1 The original BOOZ\_XFORM lineage

The classical Boozer transform implementation has evolved through several code bases:

1. **FORTRAN BOOZ\_XFORM in STELLOPT**  
   The earliest widely-used implementation lives in the STELLOPT optimisation suite as a FORTRAN code:  
   - Source: <https://github.com/PrincetonUniversity/STELLOPT/tree/develop/BOOZ_XFORM>  
   - Documentation: <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM>  

2. **Modernised C++/Python BOOZ\_XFORM (Hidden Symmetries project)**  
   A modern rewrite in C++ with a Python interface is maintained by the **Hidden Symmetries** collaboration:  
   - GitHub: <https://github.com/hiddenSymmetries/booz_xform>  
   - Theory and API docs: <https://hiddensymmetries.github.io/booz_xform/>  

   This version is highly optimised, integrated with VMEC and `simsopt`, and has a rich test suite and documentation describing the theory of the transform and its numerical implementation.

3. **This project: `booz_xform_jax`**  
   The goal here is a **JAX-native** implementation that:
   - keeps the *same physics* and nearly the same numerical algorithm as the C++ version;
   - exposes a **pure-Python**, differentiable interface;
   - can be integrated into **gradient-based optimisation and machine learning loops** (e.g., with JAX, Equinox, Optax).

This repository is not a fork of the C++ implementation but a **from-scratch re-expression** of its algorithm in JAX/NumPy, written to be both high-performance and pedagogically clear.

---

## 2. Key features

- **JAX-native numerical core**
  - All heavy computations use `jax.numpy` and vectorised operations.
  - Minimal Python loops; expensive Fourier sums are expressed as batched `einsum` / matmul-style contractions.
  - Designed so that the main transform can be wrapped in `jax.jit` or `jax.vmap` by the *user*, depending on their workflow.

- **Close correspondence to C++ BOOZ\_XFORM**
  - The implementation follows the theory and algorithm described in:
    - Hidden Symmetries BOOZ\_XFORM docs: <https://hiddensymmetries.github.io/booz_xform/theory.html>  
    - Original STELLOPT BOOZ\_XFORM docs: <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM>  
  - Mode lists, symmetry conventions, and normalisation factors are chosen to match the reference C++/Fortran behaviour as closely as possible.

- **Differentiable Boozer transform**
  - The core `BoozXform.run()` method is written in AD-friendly JAX primitives.
  - You can, in principle, differentiate \|B\| and Boozer-space diagnostics with respect to VMEC input data passed in as JAX arrays (e.g., surface-shape Fourier coefficients), enabling:
    - gradient-based stellarator optimisation,
    - embedding the transform into end-to-end differentiable pipelines (e.g., joint equilibrium + Boozer-property optimisation).

- **Pure Python I/O and plotting**
  - `vmec.py`: read VMEC `wout` files and populate the `BoozXform` instance.
  - `io_utils.py`: read/write `boozmn` files in a NetCDF format compatible with the C++ code where possible.
  - `plots.py`: quick-look diagnostics (surfplot, symplot, modeplot, wireplot) using Matplotlib.

- **Test coverage**
  - Regression tests against reference `wout_*.nc` and/or `boozmn` files.
  - Designed to slot into CI pipelines (e.g., GitHub Actions) for reproducibility.

---

## 3. Algorithm outline

The core physics is the same as in the original BOOZ\_XFORM codes. Very briefly:

1. **Read VMEC equilibrium**  
   VMEC provides Fourier coefficients for R, Z, λ (the poloidal angle shift), and the magnetic field components in VMEC coordinates on a half-grid in the radial coordinate `s`.

2. **Choose Boozer resolution**
   - Set the maximum poloidal and toroidal mode numbers `(mboz, nboz)` for the Boozer harmonics.
   - Build the Boozer mode lists `(m, n)` with the convention:
     - \(m = 0, 1, ..., m_{\text{boz}}-1\)
     - for \(m = 0\): \(n = 0, 1, ..., n_{\text{boz}}\)
     - for \(m > 0\): \(n = -n_{\text{boz}}, ..., 0, ..., n_{\text{boz}}\)

3. **VMEC → real-space fields**  
   For each selected half-grid surface:
   - Synthesize real-space fields R(θ, ζ), Z(θ, ζ), λ(θ, ζ) on a tensor-product grid in VMEC angles (θ, ζ), using the non-Nyquist VMEC Fourier coefficients.
   - Compute derivatives ∂λ/∂θ and ∂λ/∂ζ by acting with m and n factors in Fourier space.

4. **Compute auxiliary function `w` and |B|**  
   Using the Nyquist spectra of the covariant components of B, construct:
   - a scalar `w(θ, ζ)` related to the Boozer poloidal angle shift,
   - the magnetic-field strength \|B(θ, ζ)\|,
   - derivatives ∂w/∂θ and ∂w/∂ζ.

5. **Compute Boozer angles and Jacobian**  
   For each grid point:
   - Evaluate the rotational transform ι(s).
   - Form \(\nu(θ, ζ)\) as in the theory docs, then
     \[
       θ_B = θ + λ + ι \, \nu,\qquad
       ζ_B = ζ + \nu,
     \]
     along with derivatives ∂ν/∂θ, ∂ν/∂ζ.
   - Assemble the Jacobian factor and a geometric factor \(dB/d(\text{vmec})\), following the original algorithm (see the C++ docs for exact equations).

6. **Fourier transform in Boozer angles**  
   - Construct trigonometric tables in Boozer coordinates (cos m θ\_B, sin m θ\_B, cos n ζ\_B, sin n ζ\_B).
   - Perform the integrals over the grid to get:
     - Boozer-space harmonics of \|B\|,
     - Boozer coordinates R(θ\_B, ζ\_B), Z(θ\_B, ζ\_B),
     - Boozer version of the poloidal angle shift ν(θ\_B, ζ\_B),
     - Boozer Jacobian harmonics.

7. **Collect profiles and surface labels**  
   - Extract Boozer I(ψ), G(ψ) from the m=n=0 Nyquist mode.
   - Store the selected surfaces and Boozer spectra on the `BoozXform` instance for further analysis and plotting.

The JAX implementation keeps this structure but collapses many of the inner loops into high-level array operations and `einsum` contractions, which map efficiently onto CPU or GPU backends.

For further theory and detailed formulae, see:
- Hidden Symmetries BOOZ\_XFORM theory: <https://hiddensymmetries.github.io/booz_xform/theory.html>  
- STELLOPT BOOZ\_XFORM docs: <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM>  

---

## 4. Installation

### 4.1 Dependencies

- Python ≥ 3.10
- [JAX](https://github.com/google/jax) and `jaxlib`  
- NumPy
- SciPy (for some I/O or interpolation utilities)
- netCDF4 (optional, recommended for reading/writing NetCDF wout/boozmn files)
- Matplotlib (optional, for plotting)

Typical installation (CPU-only JAX):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "jax[cpu]" numpy scipy netCDF4 matplotlib
```

Then install this package in editable/development mode:

```bash
git clone https://github.com/uwplasma/boozx.git
cd boozx
pip install -e .
```

You should now be able to:

```bash
python -c "import booz_xform_jax; print(booz_xform_jax.__version__)"
```

---

## 5. Basic usage

### 5.1 From a VMEC `wout` file

The typical workflow mirrors the C++/Python BOOZ\_XFORM API:

```python
from booz_xform_jax.core import BoozXform

# Create a transform object
bx = BoozXform()

# Read VMEC wout file (NetCDF)
bx.read_wout("tests/test_files/wout_li383_1.4m.nc", flux=True)

# Optionally, register specific surfaces by index or by s-value
# - indices: 0 .. ns_in-1
bx.register_surfaces([0, 10, 20, 30])
# - or by s in [0,1]:
bx.register_surfaces(0.5)

# Run the Boozer transform
bx.run()

# Optionally save the result to a boozmn file
bx.write_boozmn("outputs/boozmn_li383_1.4m.nc")
```

After `run()`, the object `bx` contains:

- `bx.s_b`: selected surfaces (normalized toroidal flux `s`)
- `bx.xm_b, bx.xn_b`: Boozer (m, n) mode lists
- `bx.bmnc_b, bx.bmns_b`: \|B\| Fourier coefficients in Boozer coordinates
- `bx.rmnc_b, bx.rmns_b`, `bx.zmns_b, bx.zmnc_b`: geometric harmonics for R and Z
- `bx.numns_b, bx.numnc_b`: harmonics of ν
- `bx.gmnc_b, bx.gmns_b`: Boozer Jacobian harmonics
- `bx.Boozer_I, bx.Boozer_G`: Boozer I and G profiles on the selected surfaces

### 5.2 Quick-look plots

The `booz_xform_jax.plots` module reproduces some of the classic BOOZ\_XFORM plots.

```python
from booz_xform_jax.core import BoozXform
from booz_xform_jax import plots

bx = BoozXform()
bx.read_wout("tests/test_files/wout_li383_1.4m.nc", flux=True)
bx.register_surfaces([0, 10, 20, 30])
bx.run()

# |B| on a given surface in Boozer angles
plots.surfplot(bx, js=0)

# Symmetry plot: B(m,n) vs radius, grouped by symmetry class
plots.symplot(bx, max_m=20, max_n=20)

# Largest Fourier modes as a function of radius
plots.modeplot(bx, nmodes=10)

# 3D wireframe of Boozer-angle field lines on a flux surface
plots.wireplot(bx, js=-1)
```

These examples correspond closely to the documentation and figures in:
- <https://hiddensymmetries.github.io/booz_xform/usage.html>  

---

## 6. JAX, JIT, and autodiff

The `BoozXform.run()` method is written purely in JAX primitives (`jax.numpy`, `einsum`, etc.), but **the method itself is not jitted by default**. This is deliberate:

- For small or medium problems (e.g. single equilibrium, modest resolution), Python-level overhead is small and JIT compile times can dominate.
- For large or repeated runs (e.g., scanning many equilibria or surfaces), you may want to wrap the core transform in `jax.jit` or `jax.vmap` yourself, tuned to your workflow.

A typical pattern for large-scale or differentiable use-cases:

```python
import jax
from booz_xform_jax.core import BoozXform

bx = BoozXform()
bx.read_wout("tests/test_files/wout_li383_1.4m.nc", flux=True)

# Suppose you have a function that:
#   - modifies some JAX arrays on bx (e.g. rmnc, zmns),
#   - calls bx.run(),
#   - returns a scalar diagnostic based on Boozer harmonics.

def objective_from_booz(vmec_coeffs):
    # vmec_coeffs: e.g. flattened Fourier coefficients
    # 1) reshape and assign into bx.rmnc, bx.zmns, etc. (as JAX arrays)
    # 2) call bx.run() (optionally restructured into a pure function)
    # 3) build a scalar from bx.bmnc_b, bx.Boozer_I, ...
    ...
    return loss

grad_objective = jax.grad(objective_from_booz)
```

In practice, for fully end-to-end differentiable workflows, you will often wrap the core transform into a pure function:

```python
@jax.jit
def booz_transform(vmec_data, config):
    # vmec_data: pytree of JAX arrays (rmnc, zmns, etc.)
    # config: static information (nfp, mboz, nboz, grids, mode lists)
    # Returns Boozer-space spectra as JAX arrays.
    ...
    return booz_spectra
```

The current `BoozXform` class is structured so that this refactoring is straightforward: most heavy math is already in pure JAX.

---

## 7. Examples

An `examples/` directory is provided with small scripts that:

- read VMEC `wout` files from `tests/test_files` (e.g. `wout_li383_1.4m.nc`);
- run the Boozer transform on a few surfaces;
- produce quick-look plots (`surfplot`, `symplot`, `modeplot`, `wireplot`);
- optionally save Boozer spectra to `boozmn` files.

Example usage:

```bash
cd examples
python example_surfplot_li383.py
python example_symplot_li383.py
python example_wireplot_li383.py
```

Each script is self-contained and heavily commented to illustrate best practices.

---

## 8. Comparison with C++ and FORTRAN BOOZ\_XFORM

While this project aims to be numerically consistent with the C++ and FORTRAN codes, there are a few differences worth noting:

- **Language and backend**
  - C++/FORTRAN versions use explicit loops and low-level memory management for performance.
  - This JAX version expresses the same algorithm in terms of high-level array operations and lets XLA (via JAX) handle optimisation.

- **Symmetry and indexing**
  - Mode-list conventions (m, n) and radial indexing (`s_in` vs `s_b`) are chosen to match the C++ implementation as closely as possible.
  - The `register_surfaces` method mirrors the original `register` logic: you can pass either indices or continuous s values.

- **I/O formats**
  - `read_wout` and `write_boozmn` are designed to be compatible with the VMEC and BOOZ\_XFORM NetCDF conventions used in the Hidden Symmetries repo where possible, but minor differences may exist.
  - When in doubt, you can compare with the canonical C++ version:
    - <https://github.com/hiddenSymmetries/booz_xform>

- **Differentiability and JIT**
  - The C++ and FORTRAN codes were not designed for algorithmic differentiation.
  - Here, the transform is written to be AD-friendly, making Boozer-based metrics usable inside gradient-based stellarator shape optimisation and ML models.

---

## 9. Testing

To run the test suite:

```bash
pip install -e .[test]
pytest -v
```

The tests typically include:

- Regression tests that compare against reference Boozer spectra for known VMEC equilibria (e.g., `wout_li383_1.4m.nc`).
- I/O round-trip checks (`read_wout → run → write_boozmn → read_boozmn`).
- Unit tests for auxiliary helpers (mode lists, surface registration, etc.).

---

## 10. Contributing

Contributions are welcome! In particular:

- Additional tests and reference cases (e.g., QA/QH/QI stellarators).
- Performance tuning for large-resolution or GPU-heavy workflows.
- Extended plotting and diagnostic tools (e.g., bounce-averaged quantities, quasisymmetry measures).
- Clean, well-documented refactors that preserve numerical behaviour.

If you open a pull request, please:

1. Add or update tests where appropriate.
2. Run `pytest` locally before submitting.
3. Keep docstrings and comments clear, especially for numerically subtle parts of the algorithm.

---

## 11. Citation

If you use this code in scientific work, please consider citing both:

1. The **original BOOZ\_XFORM implementations**:

   - **STELLOPT / FORTRAN BOOZ\_XFORM**  
     Source and docs:  
     - <https://github.com/PrincetonUniversity/STELLOPT/tree/develop/BOOZ_XFORM>  
     - <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM>  

   - **Hidden Symmetries C++/Python BOOZ\_XFORM**  
     Source and docs:  
     - <https://github.com/hiddenSymmetries/booz_xform>  
     - <https://hiddensymmetries.github.io/booz_xform/>  

   Please refer to those repositories and docs for the canonical references.

2. This **JAX-based implementation** (`booz_xform_jax`):  
   > R. Jorge and collaborators, *booz\_xform\_jax: a JAX-based, differentiable implementation of the Boozer transform*, (in preparation).  

   A more formal reference (arXiv / journal) will be added once available.

If you build derivative codes on top of this repository, adding your own citation is strongly encouraged, and PRs that link to relevant publications are welcome.

---

## 12. License

Unless otherwise noted in individual files, this project is released under an open-source license (see `LICENSE` in the repository).

Please also respect the licenses and citation requirements of:

- VMEC,
- STELLOPT / BOOZ\_XFORM,
- Hidden Symmetries BOOZ\_XFORM,
- JAX and its dependencies.

---

## 13. Acknowledgements

This work builds conceptually and algorithmically on decades of stellarator optimisation research, including (but not limited to):

- The development of VMEC and BOOZ\_XFORM in the stellarator community.
- The modernisation of BOOZ\_XFORM and associated theory by the Hidden Symmetries collaboration.
- The broader open-source ecosystem around JAX, scientific Python, and stellarator optimisation (e.g., `simsopt`).

We are especially grateful to the authors and maintainers of:

- STELLOPT and its BOOZ\_XFORM module,  
- the C++/Python BOOZ\_XFORM repository at Hidden Symmetries,  
- and the JAX/Core Python scientific stack that makes a clean, differentiable reimplementation possible.
