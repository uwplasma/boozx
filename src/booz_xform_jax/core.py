"""
Core classes for the JAX implementation of ``booz_xform``.

This module defines the :class:`BoozXform` class, which is the primary
interface for converting Fourier data from a VMEC equilibrium
(spectral representation in VMEC coordinates) to a spectral
representation in Boozer coordinates.

Pedagogical overview
====================

**What problem are we solving?**

Given a VMEC MHD equilibrium we know (on a set of half-grid radial
surfaces) its Fourier representation in angles (θ, ζ):

  * Geometry: R(θ, ζ, s), Z(θ, ζ, s) and the poloidal angle shift
    λ(θ, ζ, s),
  * Magnetic‐field strength and covariant components:
    |B|(θ, ζ, s), B_θ(θ, ζ, s), B_ζ(θ, ζ, s).

The goal of BOOZ_XFORM is to construct a spectral representation of
the same equilibrium but in **Boozer angles** (θ_B, ζ_B), where the
magnetic field lines are straight and the contravariant components of
B take a particularly simple form. The result is stored as
Fourier coefficients B_{m,n}(s), R_{m,n}(s), Z_{m,n}(s), ν_{m,n}(s),
and Jacobian harmonics on a chosen subset of radial surfaces.

**High-level algorithm (per radial surface)**

For each selected radial surface, the core algorithm follows the
original C++ / Fortran implementation closely:

 1. Build a tensor-product grid in VMEC angles (θ, ζ) and flatten it
    to a vector of length N = N_θ × N_ζ.

 2. Using the *non-Nyquist* VMEC spectrum, synthesise:
        - R(θ, ζ), Z(θ, ζ), λ(θ, ζ),
        - ∂λ/∂θ, ∂λ/∂ζ.

 3. Using the *Nyquist* VMEC spectrum, construct:
        - an auxiliary function w(θ, ζ),
        - its derivatives ∂w/∂θ, ∂w/∂ζ,
        - |B|(θ, ζ).

 4. From the Nyquist spectra of B_θ and B_ζ, recover the Boozer
    profiles I(s) and G(s) and the auxiliary Nyquist spectrum of w.

 5. Compute the “field-line label” ν(θ, ζ) from equation (10) of the
    BOOZ_XFORM theory, and then the Boozer angles:
        θ_B = θ + λ + ι ν,
        ζ_B = ζ + ν,
    where ι is the rotational transform on this surface.

 6. From the derivatives of w and λ, construct ∂ν/∂θ and ∂ν/∂ζ and
    hence the factor dB/d(vmec) appearing in the Fourier integrals.

 7. On the (θ_B, ζ_B) grid, precompute trigonometric tables and
    perform the 2D Fourier integrals that define the Boozer
    coefficients B_{m,n}, R_{m,n}, Z_{m,n}, ν_{m,n} and the
    Boozer-Jacobian harmonics.

This module provides a **vectorised, JAX-based implementation** of the
above steps. The main performance principles are:

  * Precompute trigonometric tables on the (θ, ζ) grid once per run.
  * Hoist all per-mode cos/sin combinations that do not depend on the
    surface index out of the radial loop.
  * Replace explicit Python loops over Fourier modes by
    `jax.numpy.einsum` and broadcasting.
  * Keep the outer loop over radial surfaces in Python (a typical
    equilibrium has tens of surfaces, whereas the number of grid
    points N can be in the thousands, so most work is still inside
    JAX kernels).

Public API
==========

The external API mirrors the original BOOZ_XFORM library:

  * Create an instance of :class:`BoozXform`.
  * Call :meth:`read_wout` or :meth:`init_from_vmec` to populate
    VMEC data.
  * Optionally call :meth:`register_surfaces` to select a subset of
    radial surfaces (by index or by normalised toroidal flux s).
  * Call :meth:`run()` to perform the Boozer transform. The resulting
    Boozer spectra and profiles are stored on the instance
    (``bmnc_b``, ``rmnc_b``, ``zmns_b``, ``numns_b``, ``gmnc_b``,
    etc., plus Boozer I/G and the chosen radial grid ``s_b``).
  * Use :meth:`write_boozmn` / :meth:`read_boozmn` and plotting helpers
    (defined in other modules) as in the original code.

This file is deliberately **pedagogical**: in addition to the
performance-oriented vectorisation, it includes detailed comments
explaining each mathematical step and its relationship to the
published BOOZ_XFORM theory and to the original implementation.
"""

from __future__ import annotations

import math
import numpy as _np
from functools import partial
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

try:
    import jax
    import jax.numpy as jnp

    # The original BOOZ_XFORM (and VMEC) use double precision
    # throughout. We enable 64-bit mode globally so that JAX matches
    # the reference implementation and regression tests can compare
    # against double-precision reference outputs.
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e

from .vmec import init_from_vmec, read_wout
from .io_utils import write_boozmn, read_boozmn


# -----------------------------------------------------------------------------
# Trigonometric table helper
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(2, 3, 4))
def _init_trig(
    theta_grid: jnp.ndarray,
    zeta_grid: jnp.ndarray,
    mmax: int,
    nmax: int,
    nfp: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build trigonometric tables on a flattened (theta, zeta) grid.

    Parameters
    ----------
    theta_grid, zeta_grid :
        1D arrays of length ``n_theta_zeta`` containing the flattened
        tensor-product grid in angles. Typically these are produced by
        :meth:`BoozXform._setup_grids` via

            theta_grid = repeat(theta_vals, nzeta_full)
            zeta_grid  = tile(zeta_vals,  nu3_b)

    mmax :
        Maximum poloidal order :math:`m`.  We tabulate all m from
        0 to ``mmax`` inclusive.

    nmax :
        Maximum *period* index for :math:`n/n_fp` (i.e.
        :math:`n = k n_{fp}` with :math:`k` from 0 to ``nmax``).

    nfp :
        Number of field periods. The actual toroidal Fourier exponent
        is :math:`n \zeta = (k n_{fp}) \zeta`.

    Returns
    -------
    cosm, sinm, cosn, sinn :
        Trigonometric tables with shapes

            * ``cosm, sinm`` : (n_theta_zeta, mmax+1)
            * ``cosn, sinn`` : (n_theta_zeta, nmax+1)

        such that

            cosm[j, m] = cos(m * theta_grid[j])
            sinm[j, m] = sin(m * theta_grid[j])
            cosn[j, k] = cos(k * nfp * zeta_grid[j])
            sinn[j, k] = sin(k * nfp * zeta_grid[j])

    Notes
    -----
    In the legacy C++ code these tables were built using trigonometric
    recurrences to save flops. Here we use the direct JAX vectorised
    definitions. On modern CPUs/GPUs this is both simple and fast, and
    XLA will fuse the operations efficiently.
    """
    theta = theta_grid[:, None]  # (N, 1)
    zeta = zeta_grid[:, None]    # (N, 1)

    m_vals = jnp.arange(0, mmax + 1, dtype=jnp.float64)[None, :]  # (1, mmax+1)
    n_vals = jnp.arange(0, nmax + 1, dtype=jnp.float64)[None, :]  # (1, nmax+1)

    cosm = jnp.cos(theta * m_vals)       # (N, mmax+1)
    sinm = jnp.sin(theta * m_vals)
    cosn = jnp.cos(zeta * (n_vals * nfp))
    sinn = jnp.sin(zeta * (n_vals * nfp))

    return cosm, sinm, cosn, sinn


# -----------------------------------------------------------------------------
# Main BoozXform class
# -----------------------------------------------------------------------------


@dataclass
class BoozXform:
    """
    Class implementing the Boozer coordinate transformation using JAX.

    Instances of :class:`BoozXform` encapsulate all data required to
    convert the spectral representation of a VMEC equilibrium (in
    VMEC angles) to a spectral representation in Boozer coordinates.

    Typical usage
    -------------
    >>> bx = BoozXform()
    >>> bx.read_wout("wout_mycase.nc", flux=True)   # or init_from_vmec(...)
    >>> bx.register_surfaces([0.2, 0.5, 0.8])       # select surfaces in s-space
    >>> bx.run()
    >>> bx.write_boozmn("boozmn_mycase.nc")

    After :meth:`run` completes, the Boozer spectra are stored in
    attributes like ``bmnc_b``, ``rmnc_b``, ``zmns_b``, etc., and the
    Boozer I/G profiles and radial grid on ``Boozer_I``, ``Boozer_G``,
    and ``s_b``.

    Attributes
    ----------
    nfp : int
        Field periodicity (number of field periods) of the equilibrium.

    asym : bool
        Whether the VMEC equilibrium is non-stellarator-symmetric.
        If ``False``, only the symmetric Fourier coefficients are used
        (cosine/sine combinations that respect stellarator symmetry).
        If ``True``, additional “ns” arrays are populated and used.

    verbose : int or bool
        Controls diagnostic printing during :meth:`run`. Historically
        this was an integer (0, 1, 2, …). In this implementation any
        truthy value enables basic per-surface diagnostics; setting
        ``verbose > 1`` prints additional information.

    mpol, ntor : int
        Maximum poloidal and toroidal mode numbers in the non-Nyquist
        VMEC spectrum, read from the wout file.

    mnmax : int
        Total number of *non-Nyquist* VMEC Fourier modes. For
        symmetric equilibria this is typically ``mpol * (2*ntor + 1)``.

    xm, xn : ndarray of int, shape (mnmax,)
        Mode list for the non-Nyquist VMEC spectrum: poloidal and
        toroidal mode numbers (with xn stored as :math:`n n_{fp}` to
        match VMEC conventions).

    xm_nyq, xn_nyq : ndarray of int
        Mode list for the Nyquist spectrum used to reconstruct w and
        |B|. Sizes and ranges mirror those in the original BOOZ_XFORM.

    mpol_nyq, ntor_nyq, mnmax_nyq : int
        Nyquist resolutions and total number of Nyquist modes in the
        VMEC input, read from the wout file.

    s_in : ndarray, shape (ns_in,)
        Radial coordinate values on the VMEC half grid (excluding the
        magnetic axis). This is stored as a NumPy array (host side)
        so that we can use standard Python indexing and
        ``numpy.argmin`` when mapping floating-point s values to
        nearest indices.

    iota : jax.numpy.ndarray, shape (ns_in,)
        Rotational transform on the VMEC half grid.

    rmnc, rmns, zmnc, zmns, lmnc, lmns : jax.numpy.ndarray
        Non-Nyquist VMEC Fourier coefficients on the half grid,
        with dimensions ``(mnmax, ns_in)``. Asymmetric quantities
        are set to ``None`` when ``asym`` is ``False``.

    bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns : jax.numpy.ndarray
        Nyquist VMEC Fourier coefficients on the half grid, with
        dimensions ``(mnmax_nyq, ns_in)``. Asymmetric quantities are
        set to ``None`` when ``asym`` is ``False``. These are used to
        reconstruct |B| and the covariant components B_θ, B_ζ.

    Boozer_I_all, Boozer_G_all : ndarray, shape (ns_in,)
        Boozer I(s) and G(s) profiles on the full half grid. These
        correspond to the m=0, n=0 components of ``bsubumnc`` and
        ``bsubvmnc`` and are stored as NumPy arrays.

    phip, chi, pres, phi : jax.numpy.ndarray, shape (ns_in,), optional
        Optional radial profiles read from the VMEC file when the
        ``flux`` flag is passed to :meth:`read_wout`. They are not
        used directly in the Boozer transform but are convenient to
        have available for post-processing.

    aspect : float
        Aspect ratio of the equilibrium (copied from VMEC).

    toroidal_flux : float
        Total toroidal flux of the equilibrium (copied from VMEC).

    compute_surfs : list[int] or None
        Indices of the half-grid surfaces on which to compute the
        Boozer transform. Indices run from 0 to ``ns_in-1``.
        ``None`` (default) means “all surfaces”.

    s_b : ndarray, shape (ns_b,)
        Radial coordinate values on the subset of surfaces selected
        by ``compute_surfs``. Populated by :meth:`run` and
        :meth:`read_boozmn`.

    ns_in : int
        Number of half-grid surfaces (excluding the axis) in the VMEC
        input.

    ns_b : int
        Number of surfaces selected for the Boozer transform
        (i.e. ``len(compute_surfs)``).

    Boozer_I, Boozer_G : ndarray, shape (ns_b,)
        Boozer I and G profiles restricted to the selected surfaces.

    mboz, nboz : int
        Maximum poloidal and toroidal mode numbers in the *Boozer*
        spectrum. If not explicitly set by the user, these default to
        ``mpol`` and ``ntor`` respectively (mirroring the original
        BOOZ_XFORM behaviour).

    mnboz : int
        Total number of Boozer harmonics retained. The enumeration
        follows the original code:

          * m runs from 0, 1, …, mboz-1
          * for m = 0, n runs 0, 1, …, nboz
          * for m > 0, n runs -nboz, …, -1, 0, 1, …, nboz

        The toroidal index is stored as ``xn_b = n * nfp``.

    xm_b, xn_b : ndarray of int, shape (mnboz,)
        Boozer mode list as described above.

    bmnc_b, bmns_b, rmnc_b, rmns_b, zmnc_b, zmns_b,
    numnc_b, numns_b, gmnc_b, gmns_b : ndarray
        Boozer Fourier coefficients on the selected surfaces. Each has
        shape ``(mnboz, ns_b)``. Asymmetric arrays are ``None`` when
        ``asym`` is ``False``. The “c” suffix denotes cosine-like
        coefficients and the “s” suffix sine-like coefficients,
        following the usual VMEC/BOOZ_XFORM conventions.

    _prepared : bool
        Internal flag indicating whether the angular grids and related
        bookkeeping (θ, ζ, grid sizes) have been initialised.
    """

    # VMEC parameters read from the wout file
    nfp: int = 1
    asym: bool = False
    # Verbosity as described in the docstring
    verbose: int | bool = 1
    mpol: int = 0
    ntor: int = 0
    mnmax: int = 0
    xm: Optional[_np.ndarray] = None
    xn: Optional[_np.ndarray] = None
    xm_nyq: Optional[_np.ndarray] = None
    xn_nyq: Optional[_np.ndarray] = None
    mpol_nyq: Optional[int] = None
    ntor_nyq: Optional[int] = None
    mnmax_nyq: Optional[int] = None

    # Input arrays on the VMEC half grid (radial index runs over ns_in)
    s_in: Optional[_np.ndarray] = None
    iota: Optional[jnp.ndarray] = None
    rmnc: Optional[jnp.ndarray] = None
    rmns: Optional[jnp.ndarray] = None
    zmnc: Optional[jnp.ndarray] = None
    zmns: Optional[jnp.ndarray] = None
    lmnc: Optional[jnp.ndarray] = None
    lmns: Optional[jnp.ndarray] = None
    bmnc: Optional[jnp.ndarray] = None
    bmns: Optional[jnp.ndarray] = None
    bsubumnc: Optional[jnp.ndarray] = None
    bsubumns: Optional[jnp.ndarray] = None
    bsubvmnc: Optional[jnp.ndarray] = None
    bsubvmns: Optional[jnp.ndarray] = None
    Boozer_I_all: Optional[_np.ndarray] = None
    Boozer_G_all: Optional[_np.ndarray] = None
    phip: Optional[jnp.ndarray] = None
    chi: Optional[jnp.ndarray] = None
    pres: Optional[jnp.ndarray] = None
    phi: Optional[jnp.ndarray] = None
    aspect: float = 0.0
    toroidal_flux: float = 0.0

    # Derived quantities set by init_from_vmec or read_boozmn
    compute_surfs: Optional[List[int]] = field(default=None)
    s_b: Optional[_np.ndarray] = None
    ns_in: Optional[int] = None
    ns_b: Optional[int] = None
    Boozer_I: Optional[_np.ndarray] = None
    Boozer_G: Optional[_np.ndarray] = None
    mboz: Optional[int] = None
    nboz: Optional[int] = None
    mnboz: Optional[int] = None
    xm_b: Optional[_np.ndarray] = None
    xn_b: Optional[_np.ndarray] = None
    bmnc_b: Optional[_np.ndarray] = None
    bmns_b: Optional[_np.ndarray] = None
    rmnc_b: Optional[_np.ndarray] = None
    rmns_b: Optional[_np.ndarray] = None
    zmnc_b: Optional[_np.ndarray] = None
    zmns_b: Optional[_np.ndarray] = None
    numnc_b: Optional[_np.ndarray] = None
    numns_b: Optional[_np.ndarray] = None
    gmnc_b: Optional[_np.ndarray] = None
    gmns_b: Optional[_np.ndarray] = None

    # Bookkeeping
    _prepared: bool = False  # whether mode lists and grids have been prepared

    # ------------------------------------------------------------------
    # Delegated methods from external modules
    # ------------------------------------------------------------------

    def init_from_vmec(self, *args, s_in: Optional[_np.ndarray] = None) -> None:
        """
        Load Fourier data from VMEC into this instance.

        This method simply delegates to
        :func:`booz_xform_jax.vmec.init_from_vmec`. See that function
        for the full list of arguments and options.

        Parameters
        ----------
        *args :
            Passed directly to :func:`init_from_vmec`.
        s_in :
            Optional replacement radial grid of normalised toroidal
            flux. If provided, its first element should correspond to
            the axis; this element will be discarded so that
            ``s_in[0]`` on the instance is the first half-grid surface
            away from the axis.
        """
        init_from_vmec(self, *args, s_in=s_in)

    def read_wout(self, filename: str, flux: bool = False) -> None:
        """
        Read a VMEC ``wout`` file and populate the internal arrays.

        This is a thin wrapper around
        :func:`booz_xform_jax.vmec.read_wout`. In addition to the
        core Fourier coefficients needed for the Boozer transform,
        optional flux profile arrays can be loaded when ``flux=True``.

        Parameters
        ----------
        filename :
            Path to the VMEC wout file.
        flux :
            If ``True``, also read radial profile arrays (φ', χ, p, …).
        """
        read_wout(self, filename, flux)

    def write_boozmn(self, filename: str) -> None:
        """
        Write the computed Boozer spectra to a NetCDF file.

        This delegates to :func:`booz_xform_jax.io_utils.write_boozmn`.
        The file format (NetCDF3 vs NetCDF4) depends on the availability
        of the ``netCDF4`` package and mirrors the behaviour of the
        original BOOZ_XFORM code.
        """
        write_boozmn(self, filename)

    def read_boozmn(self, filename: str) -> None:
        """
        Read Boozer spectra from an existing ``boozmn`` file.

        This delegates to :func:`booz_xform_jax.io_utils.read_boozmn`
        and populates the current instance with the data from that file,
        including mode definitions, radial profiles, and Boozer spectra.
        """
        read_boozmn(self, filename)

    # ------------------------------------------------------------------
    # Internal helper routines for preparing mode lists and grids
    # ------------------------------------------------------------------

    def _prepare_mode_lists(self) -> None:
        """
        Construct lists of Boozer mode indices based on ``mboz`` and ``nboz``.

        The enumeration mirrors the original C++ implementation:

          * m runs from 0, 1, ..., ``mboz - 1``.
          * For m == 0, n runs 0, 1, ..., nboz (only non-negative
            toroidal indices).
          * For m > 0, n runs -nboz, ..., -1, 0, 1, ..., nboz.

        The toroidal indices are stored as ``xn_b = n * nfp`` to match
        VMEC conventions (i.e. actual Fourier angle is ``xn_b * ζ``).

        The resulting arrays are stored on ``self.xm_b`` and
        ``self.xn_b``, and the total number of modes on ``self.mnboz``.
        """
        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before preparing mode lists")

        m_list: List[int] = []
        n_list: List[int] = []

        for m in range(self.mboz):
            if m == 0:
                # m = 0 → keep only non-negative n
                for n in range(0, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)
            else:
                # m > 0 → keep full range of n
                for n in range(-self.nboz, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)

        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)
        self.mnboz = len(self.xm_b)

    def _setup_grids(self) -> None:
        """
        Set up the (theta, zeta) grid and basic bookkeeping.

        This routine constructs a tensor-product grid in VMEC angles,
        following the grid-sizing logic from the original BOOZ_XFORM
        code. The grid is slightly larger than the nominal Boozer
        resolution to comfortably resolve products of harmonics.

        For symmetric equilibria (``asym == False``) we exploit
        stellarator symmetry to restrict θ to [0, π] plus the end
        points. In that case:

          * ``ntheta_full = 2 * (2*mboz + 1)``
          * we use only the first ``nu2_b = ntheta_full//2 + 1`` rows
            in θ, i.e. 0 ≤ θ ≤ π, and apply special 1/2 weights to the
            θ=0 and θ=π rows in the Fourier integrals.

        For asymmetric equilibria (``asym == True``) we use the full
        range θ ∈ [0, 2π); then ``nu3_b = ntheta_full``.

        The flattened grids are stored on ``self._theta_grid`` and
        ``self._zeta_grid``, and grid sizes on:

          * ``self._ntheta``  – total θ points in the full grid.
          * ``self._nzeta``   – total ζ points.
          * ``self._n_theta_zeta`` – product grid size.
          * ``self._nu2_b``   – number of θ rows used in the
            symmetric case.
        """
        if self._prepared:
            return

        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before setting up grids")

        # Nominal angular resolutions (full θ range)
        ntheta_full = 2 * (2 * self.mboz + 1)
        nzeta_full = 2 * (2 * self.nboz + 1) if self.nboz > 0 else 1
        nu2_b = ntheta_full // 2 + 1  # number of θ rows in [0, π]

        if self.asym:
            # Asymmetric case: keep all θ rows in [0, 2π)
            nu3_b = ntheta_full
        else:
            # Symmetric case: exploit θ → 2π - θ symmetry, keep [0, π]
            nu3_b = nu2_b

        d_theta = (2.0 * jnp.pi) / ntheta_full
        d_zeta = (2.0 * jnp.pi) / (self.nfp * nzeta_full)

        theta_vals = jnp.arange(nu3_b) * d_theta
        zeta_vals = jnp.arange(nzeta_full) * d_zeta

        # Build flattened tensor-product grid:
        #
        #   θ_j = θ_i    for i fixed, repeated over all ζ
        #   ζ_j = ζ_k    tiled over θ rows
        #
        self._theta_grid = jnp.repeat(theta_vals, nzeta_full)
        self._zeta_grid = jnp.tile(zeta_vals, nu3_b)

        self._ntheta = int(ntheta_full)
        self._nzeta = int(nzeta_full)
        self._n_theta_zeta = int(nu3_b * nzeta_full)
        self._nu2_b = nu2_b
        self._prepared = True

    # ------------------------------------------------------------------
    # Main transform
    # ------------------------------------------------------------------

    def run(self, jit: bool = False) -> None:
        """
        Perform the Boozer coordinate transformation on selected surfaces.

        Parameters
        ----------
        jit : bool, optional
            Placeholder flag (currently unused). The transform is
            implemented entirely in terms of JAX array operations
            (``jax.numpy`` and ``einsum``). To avoid large compile
            times on CPU, we do **not** wrap the entire :meth:`run` in
            a single :func:`jax.jit` by default. Small helpers such as
            :func:`_init_trig` *are* jitted.

            Advanced users who want full JIT compilation can wrap
            :meth:`run` externally, but should be aware that this may
            lead to long compilation times for large Boozer resolutions.

        Notes
        -----
        The implementation follows the algorithm outlined in the
        module docstring and in the BOOZ_XFORM documentation. The main
        difference from a direct translation of the Fortran/C++ code is
        that all loops over Fourier modes are vectorised. Only the
        loop over radial surfaces remains as a Python loop.
        """
        _verbose = bool(self.verbose)

        if _verbose:
            print("[booz_xform_jax] Starting Boozer transform")
            print(f"[booz_xform_jax] mboz={self.mboz}, nboz={self.nboz}")

        # Basic sanity checks: VMEC data must be initialised.
        if self.rmnc is None or self.bmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")
        if self.ns_in is None:
            raise RuntimeError("ns_in must be set; did init_from_vmec run correctly?")

        ns_in = int(self.ns_in)
        if ns_in <= 0:
            raise RuntimeError("ns_in must be positive; did init_from_vmec run correctly?")

        # ------------------------------------------------------------------
        # Surface selection
        # ------------------------------------------------------------------
        # Default: compute on all surfaces.
        if self.compute_surfs is None:
            self.compute_surfs = list(range(ns_in))
        else:
            for idx in self.compute_surfs:
                if idx < 0 or idx >= ns_in:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside [0, {ns_in - 1}]"
                    )

        # ------------------------------------------------------------------
        # Boozer mode lists and grids
        # ------------------------------------------------------------------
        # Default Boozer resolution: match VMEC angular resolution.
        if self.mboz is None:
            if self.mpol is None:
                raise RuntimeError("mboz is not set and mpol is not available")
            self.mboz = int(self.mpol)

        if self.nboz is None:
            if self.ntor is None:
                raise RuntimeError("nboz is not set and ntor is not available")
            self.nboz = int(self.ntor)

        if self.mnboz is None or self.xm_b is None or self.xn_b is None:
            self._prepare_mode_lists()

        self._setup_grids()

        if _verbose:
            print("[booz_xform_jax] Grid resolution:")
            print(
                f"    ntheta={self._ntheta}, nzeta={self._nzeta},"
                f" total={self._n_theta_zeta}"
            )
            print(f"    nfp={self.nfp}, ns_b={len(self.compute_surfs)}")

        n_theta_zeta = self._n_theta_zeta
        theta_grid = self._theta_grid
        zeta_grid = self._zeta_grid

        # ------------------------------------------------------------------
        # Precompute trig tables for VMEC spectra (non-Nyquist and Nyquist)
        # and hoist all per-mode trig combinations out of the surface loop.
        # ------------------------------------------------------------------
        xm_non_np = _np.asarray(self.xm, dtype=int)
        xn_non_np = _np.asarray(self.xn, dtype=int)
        xm_nyq_np = _np.asarray(self.xm_nyq, dtype=int)
        xn_nyq_np = _np.asarray(self.xn_nyq, dtype=int)

        # Non-Nyquist (geometry, λ):
        mmax_non = int(_np.max(_np.abs(xm_non_np)))
        nmax_non = int(_np.max(_np.abs(xn_non_np // self.nfp)))
        cosm, sinm, cosn, sinn = _init_trig(
            theta_grid, zeta_grid, mmax_non, nmax_non, self.nfp
        )

        # Nyquist (w, |B|):
        mmax_nyq = int(_np.max(_np.abs(xm_nyq_np)))
        nmax_nyq = int(_np.max(_np.abs(xn_nyq_np // self.nfp)))
        cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(
            theta_grid, zeta_grid, mmax_nyq, nmax_nyq, self.nfp
        )

        # Convert mode index lists to JAX arrays once (reused per surface).
        xm_non = jnp.asarray(xm_non_np, dtype=jnp.int32)
        xn_non = jnp.asarray(xn_non_np, dtype=jnp.int32)
        xm_nyq = jnp.asarray(xm_nyq_np, dtype=jnp.int32)
        xn_nyq = jnp.asarray(xn_nyq_np, dtype=jnp.int32)

        xm_b_j = jnp.asarray(self.xm_b, dtype=jnp.int32)
        xn_b_j = jnp.asarray(self.xn_b, dtype=jnp.int32)

        # Index of (m=0, n=0) Nyquist mode → Boozer I, G.
        idx00_candidates = _np.where((xm_nyq_np == 0) & (xn_nyq_np == 0))[0]
        if len(idx00_candidates) == 0:
            raise RuntimeError("Could not find (m=0,n=0) Nyquist mode in xm_nyq/xn_nyq")
        idx00 = int(idx00_candidates[0])

        # -------------------------
        # Hoisted non-Nyquist trig combinations
        # -------------------------
        # Shapes:
        #   cosm_m_non, sinm_m_non : (N, mnmax_non)
        #   cosn_n_non, sinn_n_non : (N, mnmax_non)
        cosm_m_non = cosm[:, xm_non_np]
        sinm_m_non = sinm[:, xm_non_np]

        abs_n_non = jnp.abs(xn_non // self.nfp)
        abs_n_non_idx = _np.asarray(abs_n_non, dtype=int)
        cosn_n_non = cosn[:, abs_n_non_idx]
        sinn_n_non = sinn[:, abs_n_non_idx]

        sign_non = jnp.where(xn_non < 0, -1.0, 1.0)[None, :]

        # tcos_non / tsin_non: trigonometric factors multiplying
        # Fourier coefficients for rmnc, zmns, lmns, etc.
        tcos_non = cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non
        tsin_non = sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non

        m_non_f = xm_non.astype(jnp.float64)
        n_non_f = xn_non.astype(jnp.float64)

        # -------------------------
        # Hoisted Nyquist trig combinations
        # -------------------------
        cosm_m_nyq = cosm_nyq[:, xm_nyq_np]
        sinm_m_nyq = sinm_nyq[:, xm_nyq_np]

        abs_n_nyq = jnp.abs(xn_nyq // self.nfp)
        abs_n_nyq_idx = _np.asarray(abs_n_nyq, dtype=int)
        cosn_n_nyq = cosn_nyq[:, abs_n_nyq_idx]
        sinn_n_nyq = sinn_nyq[:, abs_n_nyq_idx]

        sign_nyq = jnp.where(xn_nyq < 0, -1.0, 1.0)[None, :]

        tcos_nyq = cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq
        tsin_nyq = sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq

        m_nyq_f = xm_nyq.astype(jnp.float64)
        n_nyq_f = xn_nyq.astype(jnp.float64)

        # ------------------------------------------------------------------
        # Output arrays (NumPy, host side)
        # ------------------------------------------------------------------
        ns_b = len(self.compute_surfs)
        self.ns_b = ns_b
        mnboz = int(self.mnboz)

        bmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
        rmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
        zmns_b = _np.zeros((mnboz, ns_b), dtype=float)
        numns_b = _np.zeros((mnboz, ns_b), dtype=float)
        gmnc_b = _np.zeros((mnboz, ns_b), dtype=float)

        if self.asym:
            bmns_b = _np.zeros((mnboz, ns_b), dtype=float)
            rmns_b = _np.zeros((mnboz, ns_b), dtype=float)
            zmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
            numnc_b = _np.zeros((mnboz, ns_b), dtype=float)
            gmns_b = _np.zeros((mnboz, ns_b), dtype=float)
        else:
            bmns_b = rmns_b = zmnc_b = numnc_b = gmns_b = None

        Boozer_I = _np.zeros(ns_b, dtype=float)
        Boozer_G = _np.zeros(ns_b, dtype=float)

        # Convenience indices for symmetric θ integration (θ=0 and θ=π rows).
        idx_theta0 = jnp.arange(0, self._nzeta)
        idx_thetapi = jnp.arange(
            (self._nu2_b - 1) * self._nzeta, self._nu2_b * self._nzeta
        )

        if _verbose:
            print(
                "                   |        outboard (theta=0)      |"
                "      inboard (theta=pi)      |"
            )
            print(
                "thread js_b js zeta| |B|input  |B|Boozer    Error   |"
                " |B|input  |B|Boozer    Error |"
            )
            print("------------------------------------------------------------------------------------")

        # ------------------------------------------------------------------
        # Loop over surfaces js_b (Python loop; heavy math is vectorised)
        # ------------------------------------------------------------------
        for js_b, js in enumerate(self.compute_surfs):
            if isinstance(self.verbose, int) and self.verbose > 1:
                print(f"[booz_xform_jax] Solving surface js_b={js_b}, js={js}")

            # ------------------------------------------------------------------
            # 1) Boozer I and G from (m=0, n=0) Nyquist mode
            # ------------------------------------------------------------------
            Boozer_I[js_b] = float(self.bsubumnc[idx00, js])
            Boozer_G[js_b] = float(self.bsubvmnc[idx00, js])

            Boozer_I_js = Boozer_I[js_b]
            Boozer_G_js = Boozer_G[js_b]

            # ------------------------------------------------------------------
            # 2) Build w spectrum (wmns, wmnc) from B_θ and B_ζ Nyquist data
            # ------------------------------------------------------------------
            # The auxiliary w(θ, ζ) is defined so that
            #
            #    ∂w/∂θ = B_ζ   (up to constants)
            #    ∂w/∂ζ = -B_θ  (up to constants)
            #
            # and its Fourier coefficients are algebraic combinations of
            # the Nyquist B_θ and B_ζ coefficients. The logic here mirrors
            # the original transpmn.f.
            bsubumnc_js = self.bsubumnc[:, js]
            bsubvmnc_js = self.bsubvmnc[:, js]
            if self.asym and self.bsubumns is not None and self.bsubvmns is not None:
                bsubumns_js = self.bsubumns[:, js]
                bsubvmns_js = self.bsubvmns[:, js]
            else:
                bsubumns_js = None
                bsubvmns_js = None

            m_nonzero = m_nyq_f != 0.0
            n_nonzero_only = jnp.logical_and(~m_nonzero, n_nyq_f != 0.0)

            # wmns: cosine-like combination
            wmns = jnp.where(
                m_nonzero,
                bsubumnc_js / m_nyq_f,
                jnp.where(n_nonzero_only, -bsubvmnc_js / n_nyq_f, 0.0),
            )
            if self.asym and bsubumns_js is not None and bsubvmns_js is not None:
                # wmnc: sine-like combination (only used in asymmetric case)
                wmnc = jnp.where(
                    m_nonzero,
                    -bsubumns_js / m_nyq_f,
                    jnp.where(n_nonzero_only, bsubvmns_js / n_nyq_f, 0.0),
                )
            else:
                wmnc = None

            bmnc_js = self.bmnc[:, js]
            bmns_js = self.bmns[:, js] if self.asym and self.bmns is not None else None

            # ------------------------------------------------------------------
            # 3) Non-Nyquist R, Z, λ and derivatives using hoisted tcos/tsin
            # ------------------------------------------------------------------
            # Grab this surface's non-Nyquist coefficients
            this_rmnc = self.rmnc[:, js]
            this_zmns = self.zmns[:, js]
            this_lmns = self.lmns[:, js]

            if self.asym and self.rmns is not None and self.zmnc is not None and self.lmnc is not None:
                this_rmns = self.rmns[:, js]
                this_zmnc = self.zmnc[:, js]
                this_lmnc = self.lmnc[:, js]
            else:
                this_rmns = this_zmnc = this_lmnc = None

            # Real-space synthesis via einsum:
            #   r(θ, ζ)   = Σ tcos_non * rmnc
            #   z(θ, ζ)   = Σ tsin_non * zmns
            #   λ(θ, ζ)   = Σ tsin_non * lmns
            #
            #   ∂λ/∂θ = Σ tcos_non * (m * lmns)
            #   ∂λ/∂ζ = -Σ tcos_non * (n * lmns)
            r = jnp.einsum("ij,j->i", tcos_non, this_rmnc)
            z = jnp.einsum("ij,j->i", tsin_non, this_zmns)
            lam = jnp.einsum("ij,j->i", tsin_non, this_lmns)
            dlam_dth = jnp.einsum("ij,j->i", tcos_non, this_lmns * m_non_f)
            dlam_dze = -jnp.einsum("ij,j->i", tcos_non, this_lmns * n_non_f)

            if self.asym and this_rmns is not None:
                r = r + jnp.einsum("ij,j->i", tsin_non, this_rmns)
                z = z + jnp.einsum("ij,j->i", tcos_non, this_zmnc)
                lam = lam + jnp.einsum("ij,j->i", tcos_non, this_lmnc)
                dlam_dth = dlam_dth - jnp.einsum(
                    "ij,j->i", tsin_non, this_lmnc * m_non_f
                )
                dlam_dze = dlam_dze + jnp.einsum(
                    "ij,j->i", tsin_non, this_lmnc * n_non_f
                )

            # ------------------------------------------------------------------
            # 4) Nyquist w, ∂w/∂θ, ∂w/∂ζ and |B| using hoisted tcos_nyq/tsin_nyq
            # ------------------------------------------------------------------
            w = jnp.einsum("ij,j->i", tsin_nyq, wmns)
            dw_dth = jnp.einsum("ij,j->i", tcos_nyq, wmns * m_nyq_f)
            dw_dze = -jnp.einsum("ij,j->i", tcos_nyq, wmns * n_nyq_f)
            bmod = jnp.einsum("ij,j->i", tcos_nyq, bmnc_js)

            if self.asym and wmnc is not None and bmns_js is not None:
                w = w + jnp.einsum("ij,j->i", tcos_nyq, wmnc)
                dw_dth = dw_dth - jnp.einsum(
                    "ij,j->i", tsin_nyq, wmnc * m_nyq_f
                )
                dw_dze = dw_dze + jnp.einsum(
                    "ij,j->i", tsin_nyq, wmnc * n_nyq_f
                )
                bmod = bmod + jnp.einsum("ij,j->i", tsin_nyq, bmns_js)

            # ------------------------------------------------------------------
            # 5) ν, Boozer angles, their derivatives, J_B, and dB/d(vmec)
            # ------------------------------------------------------------------
            this_iota = float(self.iota[js])
            GI = Boozer_G_js + this_iota * Boozer_I_js
            one_over_GI = 1.0 / GI

            # ν from eq (10): ν = (w - I λ) / (G + ι I)
            nu = one_over_GI * (w - Boozer_I_js * lam)

            # Boozer angles from eq (3):
            #   θ_B = θ + λ + ι ν
            #   ζ_B = ζ + ν
            theta_B = theta_grid + lam + this_iota * nu
            zeta_B = zeta_grid + nu

            # Derivatives of ν:
            dnu_dze = one_over_GI * (dw_dze - Boozer_I_js * dlam_dze)
            dnu_dth = one_over_GI * (dw_dth - Boozer_I_js * dlam_dth)

            # Eq (12): dB/d(vmec) factor
            dB_dvmec = (1.0 + dlam_dth) * (1.0 + dnu_dze) + \
                (this_iota - dlam_dze) * dnu_dth

            # Optional diagnostics: check |B| consistency at outboard / inboard
            if _verbose:
                B_in_ob = float(jnp.mean(bmod[idx_theta0]))
                B_bz_ob = float(jnp.mean(bmod[idx_theta0] * dB_dvmec[idx_theta0]))
                err_ob = B_bz_ob - B_in_ob

                B_in_ib = float(jnp.mean(bmod[idx_thetapi]))
                B_bz_ib = float(jnp.mean(bmod[idx_thetapi] * dB_dvmec[idx_thetapi]))
                err_ib = B_bz_ib - B_in_ib

                print(
                    f"  {js_b:4d} {js:4d}    0   "
                    f"{B_in_ob:10.6f} {B_bz_ob:10.6f} {err_ob:10.6f}   "
                    f"{B_in_ib:10.6f} {B_bz_ib:10.6f} {err_ib:10.6f}"
                )

            # ------------------------------------------------------------------
            # 6) Boozer trig tables on (theta_B, zeta_B)
            # ------------------------------------------------------------------
            # We now regard (θ_B, ζ_B) as the independent variables and
            # build trigonometric tables used in the final Fourier integrals.
            cosm_b, sinm_b, cosn_b, sinn_b = _init_trig(
                theta_B, zeta_B, int(self.mboz), int(self.nboz), self.nfp
            )

            # Symmetric θ integration: half weight for θ=0 and θ=π rows.
            # This implements the standard trapezoidal rule on [0, π]
            # when using only one half of the full θ range.
            if not self.asym:
                cosm_b = cosm_b.at[idx_theta0, :].set(cosm_b[idx_theta0, :] * 0.5)
                cosm_b = cosm_b.at[idx_thetapi, :].set(
                    cosm_b[idx_thetapi, :] * 0.5
                )
                sinm_b = sinm_b.at[idx_theta0, :].set(sinm_b[idx_theta0, :] * 0.5)
                sinm_b = sinm_b.at[idx_thetapi, :].set(
                    sinm_b[idx_thetapi, :] * 0.5
                )

            # Boozer Jacobian:
            #   J_B = (G + ι I) / |B|² = GI / |B|²
            boozer_jac = GI / (bmod * bmod)

            # ------------------------------------------------------------------
            # 7) Final Fourier integrals (all Boozer modes at once)
            # ------------------------------------------------------------------
            m_b = xm_b_j                              # (mnboz,)
            n_b = xn_b_j                              # (mnboz,)
            abs_n_b = jnp.abs(n_b // self.nfp)
            abs_n_b_idx = _np.asarray(abs_n_b, dtype=int)
            sign_b = jnp.where(n_b < 0, -1.0, 1.0)[None, :]  # (1, mnboz)

            # Gather cos/sin factors for each (m_b, n_b)
            cosm_b_m = cosm_b[:, _np.asarray(m_b, dtype=int)]      # (N, mnboz)
            sinm_b_m = sinm_b[:, _np.asarray(m_b, dtype=int)]
            cosn_b_n = cosn_b[:, abs_n_b_idx]
            sinn_b_n = sinn_b[:, abs_n_b_idx]

            # tcos / tsin as in the original code:
            tcos_modes = cosm_b_m * cosn_b_n + sinm_b_m * sinn_b_n * sign_b
            tsin_modes = sinm_b_m * cosn_b_n - cosm_b_m * sinn_b_n * sign_b

            # Fourier normalisation factor
            if self.asym:
                # Asymmetric: integrate over full θ range
                fourier_factor0 = 2.0 / (self._ntheta * self._nzeta)
            else:
                # Symmetric: integrate over [0, π]; only nu2_b-1 rows are
                # interior; θ=0 and θ=π are half-weight rows.
                fourier_factor0 = 2.0 / ((self._nu2_b - 1) * self._nzeta)

            fourier_factor = jnp.ones((mnboz,), dtype=jnp.float64) * fourier_factor0
            # Extra 1/2 for the m=0, n=0 mode (jmn=0).
            fourier_factor = fourier_factor.at[0].set(fourier_factor0 * 0.5)

            # Weight including dB/d(vmec)
            weight = dB_dvmec[:, None] * fourier_factor[None, :]  # (N, mnboz)
            tcos_w = tcos_modes * weight
            tsin_w = tsin_modes * weight

            # Integrals over all grid points for each Boozer mode:
            #   bmnc_b ∼ ∫ tcos_w * |B|
            #   rmnc_b ∼ ∫ tcos_w * R
            #   zmns_b ∼ ∫ tsin_w * Z
            #   numns_b ∼ ∫ tsin_w * ν
            #   gmnc_b ∼ ∫ tcos_w * J_B
            bmnc_b_js = jnp.einsum("ij,i->j", tcos_w, bmod)
            rmnc_b_js = jnp.einsum("ij,i->j", tcos_w, r)
            zmns_b_js = jnp.einsum("ij,i->j", tsin_w, z)
            numns_b_js = jnp.einsum("ij,i->j", tsin_w, nu)
            gmnc_b_js = jnp.einsum("ij,i->j", tcos_w, boozer_jac)

            if self.asym:
                bmns_b_js = jnp.einsum("ij,i->j", tsin_w, bmod)
                rmns_b_js = jnp.einsum("ij,i->j", tsin_w, r)
                zmnc_b_js = jnp.einsum("ij,i->j", tcos_w, z)
                numnc_b_js = jnp.einsum("ij,i->j", tcos_w, nu)
                gmns_b_js = jnp.einsum("ij,i->j", tsin_w, boozer_jac)

            # Transfer to NumPy output buffers (host side)
            bmnc_b[:, js_b] = _np.asarray(bmnc_b_js)
            rmnc_b[:, js_b] = _np.asarray(rmnc_b_js)
            zmns_b[:, js_b] = _np.asarray(zmns_b_js)
            numns_b[:, js_b] = _np.asarray(numns_b_js)
            gmnc_b[:, js_b] = _np.asarray(gmnc_b_js)

            if self.asym:
                bmns_b[:, js_b] = _np.asarray(bmns_b_js)
                rmns_b[:, js_b] = _np.asarray(rmns_b_js)
                zmnc_b[:, js_b] = _np.asarray(zmnc_b_js)
                numnc_b[:, js_b] = _np.asarray(numnc_b_js)
                gmns_b[:, js_b] = _np.asarray(gmns_b_js)

        # ------------------------------------------------------------------
        # Store results on the instance
        # ------------------------------------------------------------------
        self.bmnc_b = bmnc_b
        self.rmnc_b = rmnc_b
        self.zmns_b = zmns_b
        self.numns_b = numns_b
        self.gmnc_b = gmnc_b
        if self.asym:
            self.bmns_b = bmns_b
            self.rmns_b = rmns_b
            self.zmnc_b = zmnc_b
            self.numnc_b = numnc_b
            self.gmns_b = gmns_b

        self.Boozer_I = Boozer_I
        self.Boozer_G = Boozer_G
        self.s_b = _np.asarray(self.s_in)[self.compute_surfs]

    # ------------------------------------------------------------------
    # Surface registration (unchanged API)
    # ------------------------------------------------------------------

    def register_surfaces(self, s: Iterable[int | float] | int | float) -> None:
        """
        Register one or more surfaces on which to compute the transform.

        This method mirrors the original C++ ``register`` routine. It
        accepts either integer half-grid indices or floating-point
        radial coordinate values in normalised toroidal flux space.

        Parameters
        ----------
        s : int, float, or iterable of these
            Surfaces to register:

              * If an integer, it is interpreted as an index on the
                VMEC half grid (0 ≤ index < ns_in).
              * If a float, it should lie in [0, 1] and is interpreted
                as a normalised toroidal flux value. We then choose
                the nearest index based on ``self.s_in``.

        Notes
        -----
        * Any new surfaces are **appended** to the existing
          :attr:`compute_surfs` list (duplicates are removed).
        * Surfaces outside the valid index range produce a
          :class:`ValueError`.
        * The method does not perform the transform; you must call
          :meth:`run` afterwards.
        """
        # Normalise input to a list
        if isinstance(s, (int, float)):
            ss = [s]
        else:
            ss = list(s)

        if self.compute_surfs is None:
            current = set()
        else:
            current = set(self.compute_surfs)

        for val in ss:
            if isinstance(val, int):
                # Integer: treated as direct index
                idx = val
            else:
                # Float: map to nearest index based on s_in
                sval = float(val)
                if sval < 0.0 or sval > 1.0:
                    raise ValueError("Normalized toroidal flux values must lie in [0,1]")
                idx = int(_np.argmin(_np.abs(self.s_in - sval)))  # type: ignore[arg-type]

            if idx < 0 or idx >= int(self.ns_in):
                raise ValueError(
                    f"Surface index {idx} is outside the range [0, {int(self.ns_in) - 1}]"
                )
            current.add(idx)

        self.compute_surfs = sorted(current)
        # Respect the verbose flag: only print when truthy
        if bool(self.verbose):
            print(f"[booz_xform_jax] Registered surfaces: {self.compute_surfs}")
        return None
