"""Core classes for the JAX implementation of ``booz_xform``.

This module defines the :class:`BoozXform` class, which is the primary
interface for converting Fourier data from a VMEC equilibrium to
Boozer-coordinate Fourier data.

Compared to the reference JAX port, this version is aggressively
optimised for speed:

  * All expensive loops over Fourier modes are fully vectorised with JAX.
  * The VMEC → real-space synthesis and Nyquist ``w`` / ``|B|`` builds
    are done via batched matrix multiplications / einsums rather than
    Python loops.
  * The final Boozer-space Fourier integrals are computed for *all*
    Boozer modes at once.
  * JIT compilation is used only for small, shape-static helpers
    (trig table construction), to avoid large compile times and
    excessive recompilation.

The public API is unchanged:

  * Use :meth:`read_wout` or :meth:`init_from_vmec` to provide VMEC data.
  * Optionally call :meth:`register_surfaces` to select a subset of
    radial surfaces.
  * Call :meth:`run` to compute Boozer harmonics and scalar profiles.
  * Use :meth:`write_boozmn` / :meth:`read_boozmn` and plotting helpers
    from other modules as before.

Accuracy is preserved: the algebra matches the original algorithm and
the reference C++/Fortran implementation, including symmetry factors
for the θ integration and the treatment of the m=0, n=0 mode.
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

    # Enable double precision throughout the package.  The original
    # ``booz_xform`` code uses double precision exclusively, and the
    # regression tests compare against double precision reference
    # results.  JAX defaults to single precision on many platforms; the
    # following flag ensures that 64-bit floats are available.
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
# Trigonometric table helper (small jitted function)
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(2, 3, 4))
def _init_trig(
    theta_grid: jnp.ndarray,
    zeta_grid: jnp.ndarray,
    mmax: int,
    nmax: int,
    nfp: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analogue of the C++ ``init_trig`` function.

    Parameters
    ----------
    theta_grid, zeta_grid : (n_theta_zeta,) jnp.ndarray
        Flattened (theta, zeta) grids.
    mmax : int
        Maximum poloidal mode index m.
    nmax : int
        Maximum *period* index for n (i.e. max |n/nfp|).
    nfp : int
        Number of field periods (used in n * nfp * zeta).

    Returns
    -------
    cosm, sinm, cosn, sinn : jnp.ndarray
        Trigonometric tables with shapes
            cosm, sinm : (n_theta_zeta, mmax+1)
            cosn, sinn : (n_theta_zeta, nmax+1)
        such that:
            cosm[j, m] = cos(m * theta_grid[j])
            sinm[j, m] = sin(m * theta_grid[j])
            cosn[j, k] = cos(k * nfp * zeta_grid[j])
            sinn[j, k] = sin(k * nfp * zeta_grid[j])
    """
    theta = theta_grid[:, None]  # (n_theta_zeta, 1)
    zeta = zeta_grid[:, None]    # (n_theta_zeta, 1)

    m_vals = jnp.arange(0, mmax + 1, dtype=jnp.float64)[None, :]  # (1, mmax+1)
    n_vals = jnp.arange(0, nmax + 1, dtype=jnp.float64)[None, :]  # (1, nmax+1)

    # Direct vectorised definitions (numerically equivalent to the
    # recurrence relations in the C++ init_trig).
    cosm = jnp.cos(theta * m_vals)       # (n_theta_zeta, mmax+1)
    sinm = jnp.sin(theta * m_vals)
    cosn = jnp.cos(zeta * (n_vals * nfp))
    sinn = jnp.sin(zeta * (n_vals * nfp))

    return cosm, sinm, cosn, sinn


# -----------------------------------------------------------------------------
# Main BoozXform class
# -----------------------------------------------------------------------------


@dataclass
class BoozXform:
    """Class implementing the Boozer coordinate transformation using JAX.

    This class encapsulates all the data required to convert the spectral
    representation of a VMEC equilibrium to a spectral representation in
    Boozer coordinates.

    The implementation in this file is optimised for speed:

      * No Python loops over mode indices: all summations over Fourier
        modes are handled by JAX vector operations.
      * The only jitted helper is :func:`_init_trig`, which has tiny
        compile time and is reused across runs.
      * The main :meth:`run` method remains a normal Python method so
        that users can call it interactively without large JIT
        overheads. Heavy numerical work inside :meth:`run` is still
        carried out by JAX kernels.

    Attributes
    ----------
    (Same as in your original version; only the implementation details
    have been changed for performance.)
    """

    # VMEC parameters read from the wout file
    nfp: int = 1
    asym: bool = False
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
        """Load Fourier data from VMEC into this instance."""
        init_from_vmec(self, *args, s_in=s_in)

    def read_wout(self, filename: str, flux: bool = False) -> None:
        """Read a VMEC ``wout`` file and populate the internal arrays."""
        read_wout(self, filename, flux)

    def write_boozmn(self, filename: str) -> None:
        """Write the computed Boozer spectra to a NetCDF file."""
        write_boozmn(self, filename)

    def read_boozmn(self, filename: str) -> None:
        """Read Boozer spectra from an existing ``boozmn`` file."""
        read_boozmn(self, filename)

    # ------------------------------------------------------------------
    # Internal helper routines for preparing mode lists and grids
    # ------------------------------------------------------------------

    def _prepare_mode_lists(self) -> None:
        """Construct lists of Boozer mode indices based on ``mboz`` and ``nboz``.

        The harmonics are enumerated as in the original C++ implementation:

          * m = 0, n = 0..nboz
          * m > 0, n = -nboz..nboz

        and stored in ``xm_b`` / ``xn_b`` with ``xn_b`` in VMEC
        convention (n*nfp).
        """
        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before preparing mode lists")

        m_list: List[int] = []
        n_list: List[int] = []

        for m in range(self.mboz):
            if m == 0:
                for n in range(0, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)
            else:
                for n in range(-self.nboz, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)

        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)
        self.mnboz = len(self.xm_b)

    def _setup_grids(self) -> None:
        """Set up the (theta, zeta) grid and basic bookkeeping.

        This mirrors the grid construction in the original C++ code,
        but is used here purely with vectorised JAX operations.
        """
        if self._prepared:
            return

        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before setting up grids")

        # Full θ, ζ resolution consistent with Boozer truncation
        ntheta_full = 2 * (2 * self.mboz + 1)
        nzeta_full = 2 * (2 * self.nboz + 1) if self.nboz > 0 else 1
        nu2_b = ntheta_full // 2 + 1

        if self.asym:
            nu3_b = ntheta_full
        else:
            nu3_b = nu2_b

        d_theta = (2.0 * jnp.pi) / ntheta_full
        d_zeta = (2.0 * jnp.pi) / (self.nfp * nzeta_full)

        theta_vals = jnp.arange(nu3_b) * d_theta
        zeta_vals = jnp.arange(nzeta_full) * d_zeta

        # Flattened tensor-product grid
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

    def run(self) -> None:
        """Perform the Boozer coordinate transformation on selected surfaces.

        This method is written to minimise Python overhead while still
        being convenient to call interactively. The heavy numerical
        work is carried out by vectorised JAX operations.

        Algorithm (same as original):

          1. Build flattened (theta, zeta) grid and precompute trig
             tables for VMEC non-Nyquist and Nyquist spectra.
          2. For each selected surface:
               * Construct R, Z, λ and their derivatives in real space.
               * Construct w, ∂w/∂θ, ∂w/∂ζ and |B|.
               * Compute ν, Boozer angles (θ_B, ζ_B), and dB/d(vmec).
               * Build trig tables in Boozer coordinates.
               * Perform 2D Fourier integrals to obtain Boozer
                 harmonics and Boozer Jacobian harmonics.
        """
        _verbose = bool(self.verbose)

        if _verbose:
            print("[booz_xform_jax] Starting Boozer transform")
            print(f"[booz_xform_jax] mboz={self.mboz}, nboz={self.nboz}")

        # Basic checks
        if self.rmnc is None or self.bmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")
        if self.ns_in is None:
            raise RuntimeError("ns_in must be set; did init_from_vmec run correctly?")

        ns_in = int(self.ns_in)
        if ns_in <= 0:
            raise RuntimeError("ns_in must be positive; did init_from_vmec run correctly?")

        # Default: compute on all surfaces
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
            print(f"[booz_xform_jax] Grid resolution:")
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
        # ------------------------------------------------------------------
        xm_non_np = _np.asarray(self.xm, dtype=int)
        xn_non_np = _np.asarray(self.xn, dtype=int)
        xm_nyq_np = _np.asarray(self.xm_nyq, dtype=int)
        xn_nyq_np = _np.asarray(self.xn_nyq, dtype=int)

        mmax_non = int(_np.max(_np.abs(xm_non_np)))
        nmax_non = int(_np.max(_np.abs(xn_non_np // self.nfp)))
        cosm, sinm, cosn, sinn = _init_trig(
            theta_grid, zeta_grid, mmax_non, nmax_non, self.nfp
        )

        mmax_nyq = int(_np.max(_np.abs(xm_nyq_np)))
        nmax_nyq = int(_np.max(_np.abs(xn_nyq_np // self.nfp)))
        cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(
            theta_grid, zeta_grid, mmax_nyq, nmax_nyq, self.nfp
        )

        # Convert mode index lists to JAX arrays once (reused per surface)
        xm_non = jnp.asarray(xm_non_np, dtype=jnp.int32)
        xn_non = jnp.asarray(xn_non_np, dtype=jnp.int32)
        xm_nyq = jnp.asarray(xm_nyq_np, dtype=jnp.int32)
        xn_nyq = jnp.asarray(xn_nyq_np, dtype=jnp.int32)

        xm_b_j = jnp.asarray(self.xm_b, dtype=jnp.int32)
        xn_b_j = jnp.asarray(self.xn_b, dtype=jnp.int32)

        # Index of (m=0,n=0) Nyquist mode → Boozer I, G
        idx00_candidates = _np.where((xm_nyq_np == 0) & (xn_nyq_np == 0))[0]
        if len(idx00_candidates) == 0:
            raise RuntimeError("Could not find (m=0,n=0) Nyquist mode in xm_nyq/xn_nyq")
        idx00 = int(idx00_candidates[0])

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

        # Convenience: θ=0 and θ=π row indices for symmetric case
        if not self.asym:
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
        # Loop over surfaces js_b (Python loop; heavy math inside is vectorised)
        # ------------------------------------------------------------------
        for js_b, js in enumerate(self.compute_surfs):
            if isinstance(self.verbose, int) and self.verbose > 1:
                print(f"[booz_xform_jax] Solving surface js_b={js_b}, js={js}")

            # ------------------------------------------------------------------
            # 1) Boozer I and G (from m=n=0 Nyquist mode)
            # ------------------------------------------------------------------
            Boozer_I[js_b] = float(self.bsubumnc[idx00, js])
            Boozer_G[js_b] = float(self.bsubvmnc[idx00, js])

            Boozer_I_js = Boozer_I[js_b]
            Boozer_G_js = Boozer_G[js_b]

            # ------------------------------------------------------------------
            # 2) Build wmns / wmnc (Nyquist w spectrum) in a vectorised way
            #    and prepare Nyquist B coefficients.
            # ------------------------------------------------------------------
            m_nyq_f = xm_nyq.astype(jnp.float64)
            n_nyq_f = xn_nyq.astype(jnp.float64)

            bsubumnc_js = self.bsubumnc[:, js]
            bsubvmnc_js = self.bsubvmnc[:, js]
            if self.asym and self.bsubumns is not None and self.bsubvmns is not None:
                bsubumns_js = self.bsubumns[:, js]
                bsubvmns_js = self.bsubvmns[:, js]
            else:
                bsubumns_js = None
                bsubvmns_js = None

            # Masks for the piecewise definition of w coefficients
            m_nonzero = m_nyq_f != 0.0
            n_nonzero_only = jnp.logical_and(~m_nonzero, n_nyq_f != 0.0)

            wmns = jnp.where(
                m_nonzero,
                bsubumnc_js / m_nyq_f,
                jnp.where(n_nonzero_only, -bsubvmnc_js / n_nyq_f, 0.0),
            )
            if self.asym and bsubumns_js is not None and bsubvmns_js is not None:
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
            # 3) Non-Nyquist R, Z, λ and derivatives (vcoords_rz) – vectorised
            # ------------------------------------------------------------------
            mnmax_non = int(self.mnmax)

            this_rmnc = self.rmnc[:, js]
            this_zmns = self.zmns[:, js]
            this_lmns = self.lmns[:, js]

            if self.asym and self.rmns is not None and self.zmnc is not None and self.lmnc is not None:
                this_rmns = self.rmns[:, js]
                this_zmnc = self.zmnc[:, js]
                this_lmnc = self.lmnc[:, js]
            else:
                this_rmns = this_zmnc = this_lmnc = None

            # Shape: (n_theta_zeta, mnmax_non)
            cosm_m_non = cosm[:, xm_non_np]
            sinm_m_non = sinm[:, xm_non_np]

            abs_n_non = jnp.abs(xn_non // self.nfp)
            cosn_n_non = cosn[:, _np.asarray(abs_n_non, dtype=int)]
            sinn_n_non = sinn[:, _np.asarray(abs_n_non, dtype=int)]

            sign_non = jnp.where(xn_non < 0, -1.0, 1.0)[None, :]

            # tcos_non / tsin_non: (n_theta_zeta, mnmax_non)
            tcos_non = (
                cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non
            )
            tsin_non = (
                sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non
            )

            m_non_f = xm_non.astype(jnp.float64)
            n_non_f = xn_non.astype(jnp.float64)

            # r, z, λ, ∂λ/∂θ, ∂λ/∂ζ (all (n_theta_zeta,))
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
            # 4) Nyquist w, ∂w/∂θ, ∂w/∂ζ and |B| (vcoords_w) – vectorised
            # ------------------------------------------------------------------
            cosm_m_nyq = cosm_nyq[:, xm_nyq_np]
            sinm_m_nyq = sinm_nyq[:, xm_nyq_np]

            abs_n_nyq = jnp.abs(xn_nyq // self.nfp)
            cosn_n_nyq = cosn_nyq[:, _np.asarray(abs_n_nyq, dtype=int)]
            sinn_n_nyq = sinn_nyq[:, _np.asarray(abs_n_nyq, dtype=int)]

            sign_nyq = jnp.where(xn_nyq < 0, -1.0, 1.0)[None, :]

            tcos_nyq = (
                cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq
            )
            tsin_nyq = (
                sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq
            )

            # w, ∂w/∂θ, ∂w/∂ζ, |B|
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
            # 5) ν, Boozer angles, derivatives, J_B, dB/d(vmec)
            # ------------------------------------------------------------------
            this_iota = float(self.iota[js])
            GI = Boozer_G_js + this_iota * Boozer_I_js
            one_over_GI = 1.0 / GI

            # ν from eq (10)
            nu = one_over_GI * (w - Boozer_I_js * lam)

            # Boozer angles from eq (3)
            theta_B = theta_grid + lam + this_iota * nu
            zeta_B = zeta_grid + nu

            # Derivatives of ν
            dnu_dze = one_over_GI * (dw_dze - Boozer_I_js * dlam_dze)
            dnu_dth = one_over_GI * (dw_dth - Boozer_I_js * dlam_dth)

            # Eq (12): dB/d(vmec)
            dB_dvmec = (1.0 + dlam_dth) * (1.0 + dnu_dze) + \
                (this_iota - dlam_dze) * dnu_dth

            # Optional diagnostics: check |B| consistency at outboard/inboard
            if _verbose:
                idx_ob = jnp.arange(0, self._nzeta)
                idx_ib = jnp.arange(
                    (self._nu2_b - 1) * self._nzeta, self._nu2_b * self._nzeta
                )

                B_in_ob = float(jnp.mean(bmod[idx_ob]))
                B_bz_ob = float(jnp.mean(bmod[idx_ob] * dB_dvmec[idx_ob]))
                err_ob = B_bz_ob - B_in_ob

                B_in_ib = float(jnp.mean(bmod[idx_ib]))
                B_bz_ib = float(jnp.mean(bmod[idx_ib] * dB_dvmec[idx_ib]))
                err_ib = B_bz_ib - B_in_ib

                print(
                    f"  {js_b:4d} {js:4d}    0   "
                    f"{B_in_ob:10.6f} {B_bz_ob:10.6f} {err_ob:10.6f}   "
                    f"{B_in_ib:10.6f} {B_bz_ib:10.6f} {err_ib:10.6f}"
                )

            # ------------------------------------------------------------------
            # 6) Boozer trig tables on (theta_B, zeta_B)
            # ------------------------------------------------------------------
            cosm_b, sinm_b, cosn_b, sinn_b = _init_trig(
                theta_B, zeta_B, int(self.mboz), int(self.nboz), self.nfp
            )

            # Symmetric θ integration: half-weight θ=0 and θ=π rows
            if not self.asym:
                cosm_b = cosm_b.at[idx_theta0, :].set(cosm_b[idx_theta0, :] * 0.5)
                cosm_b = cosm_b.at[idx_thetapi, :].set(
                    cosm_b[idx_thetapi, :] * 0.5
                )
                sinm_b = sinm_b.at[idx_theta0, :].set(sinm_b[idx_theta0, :] * 0.5)
                sinm_b = sinm_b.at[idx_thetapi, :].set(
                    sinm_b[idx_thetapi, :] * 0.5
                )

            # Boozer Jacobian J_B = (G + ι I) / |B|²
            boozer_jac = GI / (bmod * bmod)

            # ------------------------------------------------------------------
            # 7) Final Fourier integrals (all Boozer modes at once)
            # ------------------------------------------------------------------
            # Gather trig factors for each Boozer mode
            m_b = xm_b_j                              # (mnboz,)
            n_b = xn_b_j                              # (mnboz,)
            abs_n_b = jnp.abs(n_b // self.nfp)
            sign_b = jnp.where(n_b < 0, -1.0, 1.0)[None, :]  # (1, mnboz)

            cosm_b_m = cosm_b[:, _np.asarray(m_b, dtype=int)]          # (Nθζ, mnboz)
            sinm_b_m = sinm_b[:, _np.asarray(m_b, dtype=int)]
            cosn_b_n = cosn_b[:, _np.asarray(abs_n_b, dtype=int)]
            sinn_b_n = sinn_b[:, _np.asarray(abs_n_b, dtype=int)]

            tcos_modes = (
                cosm_b_m * cosn_b_n + sinm_b_m * sinn_b_n * sign_b
            )  # (Nθζ, mnboz)
            tsin_modes = (
                sinm_b_m * cosn_b_n - cosm_b_m * sinn_b_n * sign_b
            )

            # Fourier normalisation factor
            if self.asym:
                # asymmetric case: integrate over the full domain
                fourier_factor0 = 2.0 / (self._ntheta * self._nzeta)
            else:
                # symmetric case: integrate over half the domain
                fourier_factor0 = 2.0 / ((self._nu2_b - 1) * self._nzeta)

            fourier_factor = jnp.ones((mnboz,), dtype=jnp.float64) * fourier_factor0

            # Extra 1/2 for the (m=0,n=0) mode, for BOTH symmetric and asymmetric
            # (this matches the original loop-based implementation)
            fourier_factor = fourier_factor.at[0].set(fourier_factor0 * 0.5)

            weight = dB_dvmec[:, None] * fourier_factor[None, :]  # (Nθζ, mnboz)
            tcos_w = tcos_modes * weight
            tsin_w = tsin_modes * weight

            # Integrals over all grid points for each Boozer mode
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

            # Transfer to NumPy output buffers
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
        """Register one or more surfaces on which to compute the transform.

        Parameters
        ----------
        s : int, float, or iterable of these
            Integer inputs are interpreted as half-grid indices on the
            VMEC radial grid. Floats in [0,1] are treated as normalised
            toroidal flux and mapped to the nearest half-grid index
            using :attr:`s_in`.
        """
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
                idx = val
            else:
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
        if bool(self.verbose):
            print(f"[booz_xform_jax] Registered surfaces: {self.compute_surfs}")
        return None
