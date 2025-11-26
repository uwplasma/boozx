"""Core classes for the JAX implementation of ``booz_xform``.

This module defines the :class:`BoozXform` class, which is the primary
interface for converting Fourier data from a VMEC equilibrium to
Boozer‐coordinate Fourier data.  The design mirrors that of the
original C++/Fortran implementation in the ``booz_xform`` project but
is implemented purely in Python and leverages JAX for vectorised
computations, just‑in‑time compilation, and automatic differentiation.

To keep the code base maintainable, the functionality is split
across several modules.  High‑level routines for reading VMEC data
live in :mod:`booz_xform_jax.vmec`, routines for reading and writing
``boozmn`` files live in :mod:`booz_xform_jax.io_utils`, and the bulk
of the numerical work is encapsulated in the :meth:`run` method here.

Users should construct a :class:`BoozXform` instance, call
:meth:`read_wout` or :meth:`init_from_vmec` to supply VMEC data,
optionally select a set of surfaces via the :attr:`compute_surfs`
property, and finally call :meth:`run` to compute the Boozer
harmonics.  The results are stored on the instance and can be
written to a NetCDF file using :meth:`write_boozmn` or passed
directly to plotting routines in :mod:`booz_xform_jax.plots`.
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
    # following flag ensures that 64‑bit floats are available.
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e

from .vmec import init_from_vmec, read_wout
from .io_utils import write_boozmn, read_boozmn

@partial(jax.jit, static_argnums=(2, 3, 4))
def _init_trig(theta_grid: jnp.ndarray,
               zeta_grid: jnp.ndarray,
               mmax: int,
               nmax: int,
               nfp: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analogue of the C++ init_trig function.

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

    # Direct vectorized definitions (numerically equivalent to the
    # recurrence relations in the C++ init_trig).
    cosm = jnp.cos(theta * m_vals)       # shape (n_theta_zeta, mmax+1)
    sinm = jnp.sin(theta * m_vals)
    cosn = jnp.cos(zeta * (n_vals * nfp))
    sinn = jnp.sin(zeta * (n_vals * nfp))

    return cosm, sinm, cosn, sinn


@dataclass
class BoozXform:
    """Class implementing the Boozer coordinate transformation using JAX.

    Instances of :class:`BoozXform` encapsulate all of the data required
    to convert the spectral representation of a VMEC equilibrium to a
    spectral representation in Boozer coordinates.  After setting up
    mode information (``mpol``, ``ntor``, ``mnmax``, etc.) and then
    populating the VMEC data via :meth:`init_from_vmec` or
    :meth:`read_wout`, call :meth:`run` to compute the Boozer
    harmonics on the requested surfaces.  The results are stored on
    the instance.

    Attributes
    ----------
    mpol : int
        Number of poloidal harmonics retained from the VMEC Fourier
        expansion.  This quantity is read from the wout file and
        should not be set manually.
    ntor : int
        Number of toroidal harmonics retained from the VMEC Fourier
        expansion.  This quantity is read from the wout file and
        should not be set manually.
    mnmax : int
        Total number of non‑Nyquist Fourier modes retained in the VMEC
        spectrum.  Equal to ``mpol*(2*ntor+1)`` for symmetric
        equilibria.
    nfp : int
        Field periodicity (number of field periods) of the equilibrium.
    asym : bool
        True if the configuration is non‑stellarator‑symmetric.  In that
        case the arrays with suffix ``_ns`` are populated.
    verbose : int
        Verbosity level.  Currently unused but present for API
        compatibility.
    compute_surfs : list[int]
        Indices of the half‑grid surfaces on which to compute the
        Boozer transform.  Indices run from 0 to ``ns_in-1`` where
        ``ns_in`` is the number of radial surfaces in the input (full
        VMEC grid minus the axis).  The default is to compute on
        all surfaces.
    s_in : ndarray
        Radial coordinate values on the VMEC half grid (excluding the
        axis).  Stored as a NumPy array so it can be indexed by
        Python lists without triggering JAX indexing restrictions.
    s_b : ndarray
        Radial coordinate values on the surfaces selected via
        ``compute_surfs``.  Populated by :meth:`run` and
        :meth:`read_boozmn`.
    iota : jax.numpy.DeviceArray
        Rotational transform on the VMEC half grid (excluding the axis).
    rmnc, rmns, zmnc, zmns, lmnc, lmns : jax.numpy.DeviceArray
        Non‑Nyquist VMEC Fourier coefficients on the half grid, with
        dimensions ``(mnmax, ns_in)``.  Asymmetric quantities are set
        to ``None`` when ``asym`` is False.
    bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns : jax.numpy.DeviceArray
        Nyquist VMEC Fourier coefficients on the half grid, with
        dimensions ``(mnmax_nyq, ns_in)``.  Asymmetric quantities are
        set to ``None`` when ``asym`` is False.
    Boozer_I_all, Boozer_G_all : ndarray
        Boozer I and G profiles on the full half grid (excluding the
        axis).  These correspond to the m=0, n=0 components of
        ``bsubumnc`` and ``bsubvmnc`` and are stored as NumPy arrays
        because they are small vectors.
    Boozer_I, Boozer_G : ndarray
        Boozer I and G on the selected surfaces.  Populated by
        :meth:`run` and :meth:`read_boozmn`.
    phip, chi, pres, phi : Optional[jax.numpy.DeviceArray]
        Optional radial profiles read from the VMEC file when the
        ``flux`` flag is passed to :meth:`read_wout`.  Each has
        dimension ``(ns_in,)``.
    xm_b, xn_b : ndarray
        Arrays of integers giving the poloidal and toroidal mode
        numbers of the Boozer harmonics, with length ``mnboz``.  The
        toroidal indices are stored as ``n*nfp`` to match VMEC
        conventions.
    bmnc_b, bmns_b, rmnc_b, rmns_b, zmnc_b, zmns_b, numnc_b, numns_b,
    gmnc_b, gmns_b : ndarray
        Boozer Fourier coefficients on the selected surfaces.  Each
        array has shape ``(mnboz, ns_b)``.  Asymmetric arrays are
        ``None`` when ``asym`` is False.
    ns_in : int
        Number of half‑grid surfaces (excluding the axis) in the VMEC
        input.
    ns_b : int
        Number of surfaces selected for the Boozer transform.
    mnboz : int
        Number of Boozer harmonics retained.  Equal to
        ``mboz*(2*nboz+1) - nboz`` due to the m=0 truncation of
        negative n.

    Notes
    -----
    The algorithm implemented here follows the description in the
    original ``booz_xform`` documentation.  The code is vectorised
    and makes extensive use of JAX to achieve performance comparable
    to the C++ implementation.  The internal arrays use a mixture
    of NumPy (for indexing convenience) and JAX arrays (for
    computation inside ``run``).  Users who wish to differentiate
    through the Boozer transform can decorate the :meth:`run` method
    with :func:`jax.jit` or :func:`jax.vmap` externally.
    """

    # VMEC parameters read from the wout file
    nfp: int = 1
    asym: bool = False
    verbose: int = 0
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

    # Delegated methods from external modules
    def init_from_vmec(self, *args, s_in: Optional[_np.ndarray] = None) -> None:
        """Load Fourier data from VMEC into this instance.

        This method delegates to :func:`booz_xform_jax.vmec.init_from_vmec`.
        See its documentation for details.  The ``s_in`` keyword
        argument, if provided, overrides the normalized toroidal flux
        grid on which the input spectra are defined.  The first
        element of ``s_in`` should correspond to the magnetic axis
        and will be discarded.
        """
        init_from_vmec(self, *args, s_in=s_in)

    def read_wout(self, filename: str, flux: bool = False) -> None:
        """Read a VMEC ``wout`` file and populate the internal arrays.

        This method delegates to :func:`booz_xform_jax.vmec.read_wout`.
        See its documentation for details.  The ``flux`` flag
        determines whether the optional flux profile arrays are read.
        """
        read_wout(self, filename, flux)

    def write_boozmn(self, filename: str) -> None:
        """Write the computed Boozer spectra to a NetCDF file.

        This method delegates to :func:`booz_xform_jax.io_utils.write_boozmn`.
        The file will be created in NetCDF4 format if the ``netCDF4``
        package is available or in NetCDF3 format otherwise.
        """
        write_boozmn(self, filename)

    def read_boozmn(self, filename: str) -> None:
        """Read Boozer spectra from an existing ``boozmn`` file.

        This method delegates to :func:`booz_xform_jax.io_utils.read_boozmn`.
        The instance will be populated with the data from the file,
        including mode definitions, radial profiles and Boozer spectra.
        """
        read_boozmn(self, filename)

    # ------------------------------------------------------------------
    # Internal helper routines for preparing mode lists and grids
    def _prepare_mode_lists(self) -> None:
        """Construct lists of Boozer mode indices based on ``mboz`` and ``nboz``.

        The Boozer harmonics are enumerated over poloidal modes m=0,...,mboz-1
        and toroidal modes n=-nboz,...,nboz with the proviso that when
        m=0 only non‑negative n are retained.  Each (m,n) pair is
        converted into a single index for accessing the output Fourier
        arrays.  The resulting arrays ``xm_b`` and ``xn_b`` have length
        ``mnboz`` and contain the poloidal and toroidal indices for each
        mode.  The toroidal index ``xn_b`` is stored as ``n*nfp`` to
        match the VMEC conventions.
          - m runs from 0, 1, ..., mboz-1
          - for m = 0, n runs 0, 1, ..., nboz
          - for m > 0, n runs -nboz, ..., -1, 0, 1, ..., nboz
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
                # m > 0 → keep all n from -nboz..nboz
                for n in range(-self.nboz, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)

        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)
        self.mnboz = len(self.xm_b)

    def _setup_grids(self) -> None:
        """Set up the (theta, zeta) grid and basic bookkeeping.

        This routine prepares several arrays needed for the coordinate
        transformation.  It computes the number of theta and zeta grid
        points based on the maximum VMEC mode numbers and the desired
        Boozer resolution ``mboz`` and ``nboz``.  It precomputes
        matrices of cosines and sines used to synthesise real‑space
        functions from spectral coefficients, as well as derivative
        factors.  The results are cached on ``self`` for reuse if
        :meth:`run` is called multiple times.

        This mirrors the grid construction in the C++ code: we build a
        tensor-product grid in (theta, zeta), then flatten it to obtain
        arrays of length n_theta_zeta.  The exact resolutions ntheta and
        nzeta are chosen based on the maximum Nyquist mode numbers,
        similarly to the original implementation.
        """
        # Skip if grids already set up
        if self._prepared:
            return
        # Ensure mboz and nboz are set
        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before setting up grids")

        ntheta_full = 2 * (2 * self.mboz + 1)
        nzeta_full  = 2 * (2 * self.nboz + 1) if self.nboz > 0 else 1
        nu2_b       = ntheta_full // 2 + 1

        if self.asym:
            nu3_b = ntheta_full
        else:
            nu3_b = nu2_b

        d_theta = (2.0 * jnp.pi) / ntheta_full
        d_zeta  = (2.0 * jnp.pi) / (self.nfp * nzeta_full)

        theta_vals = jnp.arange(nu3_b) * d_theta
        zeta_vals  = jnp.arange(nzeta_full) * d_zeta

        self._theta_grid = jnp.repeat(theta_vals, nzeta_full)
        self._zeta_grid  = jnp.tile(zeta_vals, nu3_b)

        self._ntheta         = int(ntheta_full)
        self._nzeta          = int(nzeta_full)
        self._n_theta_zeta   = int(nu3_b * nzeta_full)
        self._nu2_b          = nu2_b
        self._prepared       = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Perform the Boozer coordinate transformation on selected surfaces.

        This implementation closely follows the original C++
        ``Booz_xform::surface_solve`` algorithm:

          1. Build the flattened (theta, zeta) grid and precompute
             trig tables for VMEC (non-Nyquist) and Nyquist spectra.
          2. For each requested surface js_b (with input index js):
               - Construct R, Z, lambda and their derivatives in real
                 space on the (theta, zeta) grid.
               - Construct the auxiliary function w and |B|.
               - Compute nu, Boozer angles theta_B, zeta_B, and
                 d_Boozer_d_vmec.
               - Build trig tables in Boozer coordinates.
               - Perform the 2D Fourier integrals to obtain Boozer
                 Fourier coefficients B_{m,n}, R_{m,n}, Z_{m,n}, nu_{m,n},
                 and the Boozer Jacobian harmonics.
        """
        if self.rmnc is None or self.bmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")

        # Number of half-grid surfaces:
        ns_in = int(self.ns_in)
        if ns_in <= 0:
            raise RuntimeError("ns_in must be positive; did init_from_vmec run correctly?")

        # Default: compute on all surfaces:
        if self.compute_surfs is None:
            self.compute_surfs = list(range(ns_in))
        else:
            for idx in self.compute_surfs:
                if idx < 0 or idx >= ns_in:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside [0, {ns_in - 1}]"
                    )

        # ------------------------------------------------------------------
        # Ensure Boozer mode lists (xm_b, xn_b, mnboz) are prepared.
        # Use VMEC resolution as default if user did not set mboz/nboz.
        # ------------------------------------------------------------------
        # Default Boozer resolution if not set by user
        if self.mboz is None:
            if self.mpol is None:
                raise RuntimeError("mboz is not set and mpol is not available")
            self.mboz = int(self.mpol)
        if self.nboz is None:
            if self.ntor is None:
                raise RuntimeError("nboz is not set and ntor is not available")
            self.nboz = int(self.ntor)  # allow nboz = 0 when ntor = 0
        # Prepare mode lists
        if self.mnboz is None or self.xm_b is None or self.xn_b is None:
            self._prepare_mode_lists()
        # Set up grids
        self._setup_grids()
        ntheta = self._ntheta
        nzeta = self._nzeta
        n_theta_zeta = self._n_theta_zeta
        theta_grid = self._theta_grid
        zeta_grid = self._zeta_grid
        nu2_b = self._nu2_b  # 1-based index in the C++ code

        # Precompute trig tables for the VMEC (non-Nyquist) spectrum:
        mmax_non = int(_np.max(_np.abs(self.xm)))
        nmax_non = int(_np.max(_np.abs(self.xn // self.nfp)))
        cosm, sinm, cosn, sinn = _init_trig(theta_grid, zeta_grid,
                                            mmax_non, nmax_non, self.nfp)

        # Precompute trig tables for the Nyquist spectrum:
        mmax_nyq = int(_np.max(_np.abs(self.xm_nyq)))
        nmax_nyq = int(_np.max(_np.abs(self.xn_nyq // self.nfp)))
        cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(theta_grid, zeta_grid,
                                                            mmax_nyq, nmax_nyq, self.nfp)

        # Prepare output arrays in Boozer space:
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

        # Boozer I and G on selected surfaces:
        Boozer_I = _np.zeros(ns_b, dtype=float)
        Boozer_G = _np.zeros(ns_b, dtype=float)

        # Convenience views:
        xm_non = jnp.asarray(self.xm, dtype=jnp.int32)
        xn_non = jnp.asarray(self.xn, dtype=jnp.int32)
        xm_nyq_arr = jnp.asarray(self.xm_nyq, dtype=jnp.int32)
        xn_nyq_arr = jnp.asarray(self.xn_nyq, dtype=jnp.int32)

        # Loop over surfaces js_b:
        for js_b, js in enumerate(self.compute_surfs):
            # ---------------------------
            # Work arrays (length n_theta_zeta)
            # ---------------------------
            r = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            z = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            lam = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dlam_dth = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dlam_dze = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            w = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dw_dth = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dw_dze = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            bmod = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            nu = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dnu_dth = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dnu_dze = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            theta_B = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            zeta_B = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            dB_dvmec = jnp.zeros(n_theta_zeta, dtype=jnp.float64)
            boozer_jac = jnp.zeros(n_theta_zeta, dtype=jnp.float64)

            # wmns, wmnc (Nyquist, length mnmax_nyq)
            mnmax_nyq = int(self.mnmax_nyq)
            wmns = jnp.zeros(mnmax_nyq, dtype=jnp.float64)
            wmnc = jnp.zeros(mnmax_nyq, dtype=jnp.float64) if self.asym else None

            # ---------------------------
            # transpmn.f part: build wmns/wmnc and Boozer_I/G
            # ---------------------------
            for jmn in range(mnmax_nyq):
                m = xm_nyq_arr[jmn]
                n = xn_nyq_arr[jmn]

                if m != 0:
                    wmns = wmns.at[jmn].set(self.bsubumnc[jmn, js] / m)
                    if self.asym and self.bsubumns is not None:
                        wmnc = wmnc.at[jmn].set(-self.bsubumns[jmn, js] / m)
                elif n != 0:
                    wmns = wmns.at[jmn].set(-self.bsubvmnc[jmn, js] / n)
                    if self.asym and self.bsubvmns is not None:
                        wmnc = wmnc.at[jmn].set(self.bsubvmns[jmn, js] / n)
                else:
                    # m = n = 0 → Boozer I and G
                    Boozer_I[js_b] = float(self.bsubumnc[jmn, js])
                    Boozer_G[js_b] = float(self.bsubvmnc[jmn, js])
                    # Corresponding p mode is set to 0 by convention.

            # ---------------------------
            # vcoords_rz part: R, Z, lambda and derivatives (non-Nyquist)
            # ---------------------------
            mnmax_non = int(self.mnmax)
            # print(f"  zmns.shape={self.zmns.shape if self.zmns is not None else None}")
            this_rmnc = self.rmnc[:, js]
            this_zmns = self.zmns[:, js]
            this_lmns = self.lmns[:, js]
            this_rmns = self.rmns[:, js] if self.asym and self.rmns is not None else None
            this_zmnc = self.zmnc[:, js] if self.asym and self.zmnc is not None else None
            this_lmnc = self.lmnc[:, js] if self.asym and self.lmnc is not None else None

            for jmn in range(mnmax_non):
                m = xm_non[jmn]
                n = xn_non[jmn]
                abs_n = abs(n // self.nfp)
                sign = -1 if n < 0 else 1

                tcos = (cosm[:, m] * cosn[:, abs_n] +
                        sinm[:, m] * sinn[:, abs_n] * sign)
                tsin = (sinm[:, m] * cosn[:, abs_n] -
                        cosm[:, m] * sinn[:, abs_n] * sign)

                r = r + tcos * this_rmnc[jmn]
                z = z + tsin * this_zmns[jmn]
                lam = lam + tsin * this_lmns[jmn]
                dlam_dth = dlam_dth + tcos * this_lmns[jmn] * m
                dlam_dze = dlam_dze - tcos * this_lmns[jmn] * n

                if self.asym and this_rmns is not None:
                    r = r + tsin * this_rmns[jmn]
                    z = z + tcos * this_zmnc[jmn]
                    lam = lam + tcos * this_lmnc[jmn]
                    dlam_dth = dlam_dth - tsin * this_lmnc[jmn] * m
                    dlam_dze = dlam_dze + tsin * this_lmnc[jmn] * n

            # ---------------------------
            # vcoords_w part: w, dw/dθ, dw/dζ, |B| (Nyquist)
            # ---------------------------
            for jmn in range(mnmax_nyq):
                m = xm_nyq_arr[jmn]
                n = xn_nyq_arr[jmn]
                abs_n = abs(n // self.nfp)
                sign = -1 if n < 0 else 1

                tcos = (cosm_nyq[:, m] * cosn_nyq[:, abs_n] +
                        sinm_nyq[:, m] * sinn_nyq[:, abs_n] * sign)
                tsin = (sinm_nyq[:, m] * cosn_nyq[:, abs_n] -
                        cosm_nyq[:, m] * sinn_nyq[:, abs_n] * sign)

                w = w + tsin * wmns[jmn]
                dw_dth = dw_dth + tcos * wmns[jmn] * m
                dw_dze = dw_dze - tcos * wmns[jmn] * n
                bmod = bmod + tcos * self.bmnc[jmn, js]

                if self.asym and wmnc is not None and self.bmns is not None:
                    w = w + tcos * wmnc[jmn]
                    dw_dth = dw_dth - tsin * wmnc[jmn] * m
                    dw_dze = dw_dze + tsin * wmnc[jmn] * n
                    bmod = bmod + tsin * self.bmns[jmn, js]

            # ---------------------------
            # harfun / eq (10–12): nu, Boozer angles, d_Boozer_d_vmec
            # ---------------------------
            this_iota = float(self.iota[js])
            GI = Boozer_G[js_b] + this_iota * Boozer_I[js_b]
            one_over_GI = 1.0 / GI

            # nu from eq (10):
            nu = one_over_GI * (w - Boozer_I[js_b] * lam)

            # Boozer angles from eq (3):
            theta_B = theta_grid + lam + this_iota * nu
            zeta_B = zeta_grid + nu

            # Derivatives of nu:
            dnu_dze = one_over_GI * (dw_dze - Boozer_I[js_b] * dlam_dze)
            dnu_dth = one_over_GI * (dw_dth - Boozer_I[js_b] * dlam_dth)

            # Eq (12):
            dB_dvmec = (1.0 + dlam_dth) * (1.0 + dnu_dze) + \
                       (this_iota - dlam_dze) * dnu_dth

            # ---------------------------
            # Boozer trig tables on (theta_B, zeta_B)
            # ---------------------------
            cosm_b, sinm_b, cosn_b, sinn_b = _init_trig(theta_B, zeta_B,
                                                        int(self.mboz),
                                                        int(self.nboz),
                                                        self.nfp)

            # Symmetric θ integration factor (only half-circle in θ):
            # Apply the ½ factor for theta=0 and theta=π rows:
            if not self.asym:
                idx0   = jnp.arange(0, self._nzeta)                        # θ=0 row
                idx_pi = jnp.arange(self._nzeta*(self._nu2_b - 1), self._nzeta*self._nu2_b)  # θ=π row
                for m in range(self.mboz+1):
                    cosm_b = cosm_b.at[idx0, m].set(cosm_b[idx0, m] * 0.5)
                    sinm_b = sinm_b.at[idx0, m].set(sinm_b[idx0, m] * 0.5)
                    cosm_b = cosm_b.at[idx_pi, m].set(cosm_b[idx_pi, m] * 0.5)
                    sinm_b = sinm_b.at[idx_pi, m].set(sinm_b[idx_pi, m] * 0.5)

            # Boozer Jacobian J_B = (G + iota * I) / |B|^2
            boozer_jac = GI / (bmod * bmod)

            # ---------------------------
            # Final Fourier integrals (eq 11)
            # ---------------------------
            if self.asym:
                # asymmetric case: integrate over the full domain
                fourier_factor0 = 2.0 / (self._ntheta * self._nzeta)
            else:
                # symmetric case: integrate over half the domain
                fourier_factor0 = 2.0 / ((self._nu2_b - 1) * self._nzeta)
                # m=0 mode needs an extra factor of ½:
                # this is handled by multiplying fourier_factor by ½ for jmn==0

            for jmn in range(mnboz):
                m = int(self.xm_b[jmn])
                n = int(self.xn_b[jmn])
                abs_n = abs(n // self.nfp)
                sign = -1 if n < 0 else 1

                fourier_factor = fourier_factor0
                if jmn == 0:
                    fourier_factor *= 0.5  # extra 1/2 for m=0 mode

                # tcos / tsin as in C++:
                tcos = (cosm_b[:, m] * cosn_b[:, abs_n] +
                        sinm_b[:, m] * sinn_b[:, abs_n] * sign) \
                       * dB_dvmec * fourier_factor
                tsin = (sinm_b[:, m] * cosn_b[:, abs_n] -
                        cosm_b[:, m] * sinn_b[:, abs_n] * sign) \
                       * dB_dvmec * fourier_factor

                # Integrate over all grid points:
                bmnc_b[jmn, js_b] += float(jnp.sum(tcos * bmod))
                rmnc_b[jmn, js_b] += float(jnp.sum(tcos * r))
                zmns_b[jmn, js_b] += float(jnp.sum(tsin * z))
                numns_b[jmn, js_b] += float(jnp.sum(tsin * nu))
                gmnc_b[jmn, js_b] += float(jnp.sum(tcos * boozer_jac))

                if self.asym:
                    bmns_b[jmn, js_b] += float(jnp.sum(tsin * bmod))
                    rmns_b[jmn, js_b] += float(jnp.sum(tsin * r))
                    zmnc_b[jmn, js_b] += float(jnp.sum(tcos * z))
                    numnc_b[jmn, js_b] += float(jnp.sum(tcos * nu))
                    gmns_b[jmn, js_b] += float(jnp.sum(tsin * boozer_jac))

            print(f"Processing surface {js} (js_b={js_b}): mnmax_non={mnmax_non}, mnmax_nyq={mnmax_nyq}" + 
                  ", Boozer I = {Boozer_I[js_b]}, G = {Boozer_G[js_b]}, grid size: ntheta={ntheta}, nzeta={nzeta}" +
                  f", n_theta_zeta={n_theta_zeta}, wmns[1:3]={wmns[1:3]}, lmns.shape={self.lmns.shape if self.lmns is not None else None}" +
                  f"  rmnc.shape={self.rmnc.shape if self.rmnc is not None else None}"+
                  f"  bmnc.shape={self.bmnc.shape if self.bmnc is not None else None}"+
                  f"  bmod[1:3]={bmod[1:3]}, nu[1:3]={nu[1:3]}"
                  f"  theta_B[1:3]={theta_B[1:3]}, zeta_B[1:3]={zeta_B[1:3]}"
                  f"  dB_dvmec[1:3]={dB_dvmec[1:3]}, fourier_factor={fourier_factor}"
            )

        # Store results on the instance:
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
    def register_surfaces(self, s: Iterable[int | float] | int | float) -> None:
        """Register one or more surfaces on which to compute the transform.

        This method mirrors the ``register`` routine from the original
        C++ code.  It accepts either integer half‑grid indices or
        floating‑point radial coordinate values.  In the latter case the
        closest index on ``s_in`` is chosen.  Any new surfaces are
        appended to the existing :attr:`compute_surfs` list.  Surfaces
        outside the valid range ``[0, ns_in-1]`` raise a
        :class:`ValueError`.

        Parameters
        ----------
        s : int, float, or iterable of these
            The surfaces to register.  Integers are interpreted as
            indices on the VMEC half grid.  Floats should lie in
            ``[0,1]`` and represent normalized toroidal flux; they are
            mapped to the nearest half‑grid index.
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
                idx = val
            else:
                # Float: map to nearest index based on s_in
                sval = float(val)
                if sval < 0.0 or sval > 1.0:
                    raise ValueError("Normalized toroidal flux values must lie in [0,1]")
                idx = int(_np.argmin(_np.abs(self.s_in - sval)))
            if idx < 0 or idx >= int(self.ns_in):
                raise ValueError(
                    f"Surface index {idx} is outside the range [0, {int(self.ns_in) - 1}]"
                )
            current.add(idx)
        self.compute_surfs = sorted(current)
        return None