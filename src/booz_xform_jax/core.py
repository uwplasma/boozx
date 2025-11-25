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
    from jax.config import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e

from .vmec import init_from_vmec, read_wout
from .io_utils import write_boozmn, read_boozmn


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
        """
        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before preparing mode lists")
        m_list: List[int] = []
        n_list: List[int] = []
        for m in range(self.mboz):
            nmin = 0 if m == 0 else -self.nboz
            for n in range(nmin, self.nboz + 1):
                m_list.append(m)
                n_list.append(n * self.nfp)
        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)
        self.mnboz = len(self.xm_b)

    def _setup_grids(self) -> None:
        """Set up real‐space grids and precompute trigonometric tables.

        This routine prepares several arrays needed for the coordinate
        transformation.  It computes the number of theta and zeta grid
        points based on the maximum VMEC mode numbers and the desired
        Boozer resolution ``mboz`` and ``nboz``.  It precomputes
        matrices of cosines and sines used to synthesise real‑space
        functions from spectral coefficients, as well as derivative
        factors.  The results are cached on ``self`` for reuse if
        :meth:`run` is called multiple times.
        """
        # Skip if grids already set up
        if self._prepared:
            return
        # Determine Nyquist wavenumbers
        assert self.mnmax_nyq is not None
        assert self.xm_nyq is not None
        assert self.xn_nyq is not None
        # The Nyquist numbers include the negative toroidal modes, so
        # find the maximum absolute toroidal mode number
        mtot = int(_np.max(_np.abs(self.xm_nyq)))
        ntot = int(_np.max(_np.abs(self.xn_nyq // self.nfp)))
        # Choose resolution to be at least twice the maximum wavenumber
        # plus a factor from the desired Boozer resolution.  These
        # choices follow the original C++ code which doubles the
        # resolution to avoid aliasing.
        ntheta = max(2 * (mtot + self.mboz), 64)
        nzeta = max(2 * (ntot + self.nboz), 64)
        # Uniform grids on [0,2π)
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0.0, 2.0 * jnp.pi, nzeta, endpoint=False)
        # Precompute cos and sin matrices for non‑Nyquist and Nyquist modes
        # Non‑Nyquist (mnmax) shapes: (mnmax, ntheta)
        cos_mn = jnp.cos(jnp.outer(self.xm, theta) - jnp.outer(self.xn // self.nfp, zeta))
        sin_mn = jnp.sin(jnp.outer(self.xm, theta) - jnp.outer(self.xn // self.nfp, zeta))
        # Nyquist shapes: (mnmax_nyq, ntheta)
        cos_mn_nyq = jnp.cos(jnp.outer(self.xm_nyq, theta) - jnp.outer(self.xn_nyq // self.nfp, zeta))
        sin_mn_nyq = jnp.sin(jnp.outer(self.xm_nyq, theta) - jnp.outer(self.xn_nyq // self.nfp, zeta))
        # Derivative factors: derivative with respect to theta gives  m, with respect to zeta gives n/nfp
        dth_mn = jnp.asarray(self.xm, dtype=jnp.float64)
        dth_mn_nyq = jnp.asarray(self.xm_nyq, dtype=jnp.float64)
        dze_mn = jnp.asarray(self.xn // self.nfp, dtype=jnp.float64)
        dze_mn_nyq = jnp.asarray(self.xn_nyq // self.nfp, dtype=jnp.float64)
        # Store grids and derivative factors
        self._ntheta = ntheta
        self._nzeta = nzeta
        self._theta = theta
        self._zeta = zeta
        self._cos_mn = cos_mn
        self._sin_mn = sin_mn
        self._cos_mn_nyq = cos_mn_nyq
        self._sin_mn_nyq = sin_mn_nyq
        self._dth_mn = dth_mn
        self._dth_mn_nyq = dth_mn_nyq
        self._dze_mn = dze_mn
        self._dze_mn_nyq = dze_mn_nyq
        self._prepared = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Perform the Boozer coordinate transformation on selected surfaces.

        This method constructs real‑space representations of the magnetic
        field on each selected surface, computes the Boozer angles
        (vartheta and vphi) by solving a set of nonlinear equations, and
        finally extracts the Boozer Fourier spectra.  The computation is
        vectorised where possible and uses JAX JIT compilation for
        efficiency.  After running, the instance will have its
        Boozer spectra arrays (``bmnc_b``, ``bmns_b``, etc.) populated
        as NumPy arrays of shape ``(mnboz, ns_b)``.  The radial
        profiles ``Boozer_I`` and ``Boozer_G`` are also computed on
        the selected surfaces.
        """
        if self.rmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")
        # Set default compute_surfs if not specified
        ns_in = int(self.ns_in)
        if self.compute_surfs is None:
            self.compute_surfs = list(range(ns_in))
        else:
            # Validate indices
            for idx in self.compute_surfs:
                if idx < 0 or idx >= ns_in:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside the range [0, {ns_in - 1}]"
                    )
        # Prepare mode lists and grids if not already done
        # Set default Boozer resolution if unset.  Use VMEC resolution.
        if self.mboz is None:
            self.mboz = self.mpol
        if self.nboz is None:
            self.nboz = self.ntor
        if self.mnboz is None or self.xm_b is None or self.xn_b is None:
            self._prepare_mode_lists()
        self._setup_grids()
        # Number of surfaces to compute
        ns_b = len(self.compute_surfs)
        self.ns_b = ns_b
        # Allocate output arrays.  Use NumPy to avoid JAX indexing restrictions.
        mnboz = self.mnboz
        bmnc_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        rmnc_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        zmns_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        numns_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        gmnc_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        # Allocate asymmetric arrays only if needed
        if self.asym:
            bmns_b_out = _np.zeros((mnboz, ns_b), dtype=float)
            rmns_b_out = _np.zeros((mnboz, ns_b), dtype=float)
            zmnc_b_out = _np.zeros((mnboz, ns_b), dtype=float)
            numnc_b_out = _np.zeros((mnboz, ns_b), dtype=float)
            gmns_b_out = _np.zeros((mnboz, ns_b), dtype=float)
        else:
            bmns_b_out = None
            rmns_b_out = None
            zmnc_b_out = None
            numnc_b_out = None
            gmns_b_out = None
        # Allocate radial profiles on selected surfaces
        Boozer_I = _np.zeros(ns_b, dtype=float)
        Boozer_G = _np.zeros(ns_b, dtype=float)
        # Helper function to solve the surface equations for one surface.
        # This function is defined here so that it can close over self
        # without making everything static.  It will be JIT compiled by
        # JAX.
        def surface_solve(rmnc_j, rmns_j, zmnc_j, zmns_j, lmnc_j, lmns_j,
                          bmnc_j, bmns_j, bsubumnc_j, bsubumns_j,
                          bsubvmnc_j, bsubvmns_j, asym_flag):
            """Compute Boozer spectra on a single surface.

            Parameters
            ----------
            rmnc_j, rmns_j, zmnc_j, zmns_j : jax.numpy.DeviceArray
                VMEC Fourier coefficients of R, Z on this surface.
            lmnc_j, lmns_j : jax.numpy.DeviceArray
                VMEC Fourier coefficients of λ (lambda) on this surface.
            bmnc_j, bmns_j : jax.numpy.DeviceArray
                VMEC Fourier coefficients of the magnetic field magnitude B
                on this surface.
            bsubumnc_j, bsubumns_j, bsubvmnc_j, bsubvmns_j : jax.numpy.DeviceArray
                VMEC Fourier coefficients of the covariant components of B.
            asym_flag : bool
                Whether the equilibrium is non‑symmetric; determines
                whether sine/sine terms are used.

            Returns
            -------
            bmnc_b_s, bmns_b_s, rmnc_b_s, rmns_b_s, zmnc_b_s, zmns_b_s,
            numnc_b_s, numns_b_s, gmnc_b_s, gmns_b_s : jax.numpy.DeviceArray
                Arrays of length ``mnboz`` containing the Boozer Fourier
                coefficients for this surface.  Asymmetric arrays are
                returned only when ``asym_flag`` is True.
            """
            # Build real‑space functions using Fourier synthesis
            # R, Z, λ on grid
            r = jnp.dot(rmnc_j, self._cos_mn) - (jnp.dot(rmns_j, self._sin_mn) if asym_flag else 0.0)
            z = jnp.dot(zmns_j, self._sin_mn) + (jnp.dot(zmnc_j, self._cos_mn) if asym_flag else 0.0)
            lam = jnp.dot(lmns_j, self._sin_mn) + (jnp.dot(lmnc_j, self._cos_mn) if asym_flag else 0.0)
            # B components on grid
            Bm = jnp.dot(bmnc_j, self._cos_mn_nyq) - (jnp.dot(bmns_j, self._sin_mn_nyq) if asym_flag else 0.0)
            # Covariant components on grid
            bsubu = jnp.dot(bsubumnc_j, self._cos_mn_nyq) - (jnp.dot(bsubumns_j, self._sin_mn_nyq) if asym_flag else 0.0)
            bsubv = jnp.dot(bsubvmnc_j, self._cos_mn_nyq) - (jnp.dot(bsubvmns_j, self._sin_mn_nyq) if asym_flag else 0.0)
            # Derivatives of R, Z, λ with respect to theta and zeta
            dth_r = jnp.dot(rmnc_j * self._dth_mn, self._sin_mn) + (jnp.dot(rmns_j * self._dth_mn, self._cos_mn) if asym_flag else 0.0)
            dth_z = jnp.dot(zmns_j * self._dth_mn, self._cos_mn) - (jnp.dot(zmnc_j * self._dth_mn, self._sin_mn) if asym_flag else 0.0)
            dth_l = jnp.dot(lmns_j * self._dth_mn, self._cos_mn) - (jnp.dot(lmnc_j * self._dth_mn, self._sin_mn) if asym_flag else 0.0)
            dze_r = -jnp.dot(rmnc_j * self._dze_mn, self._sin_mn) - (jnp.dot(rmns_j * self._dze_mn, self._cos_mn) if asym_flag else 0.0)
            dze_z = jnp.dot(zmns_j * self._dze_mn, self._cos_mn) + (jnp.dot(zmnc_j * self._dze_mn, self._sin_mn) if asym_flag else 0.0)
            dze_l = jnp.dot(lmns_j * self._dze_mn, self._cos_mn) + (jnp.dot(lmnc_j * self._dze_mn, self._sin_mn) if asym_flag else 0.0)
            # Metric factors: w = sqrt((dth_r * dze_z - dze_r * dth_z)^2 + ... )
            # Compute cross products for Jacobian determinant of (r, z, φ) -> (θ, ζ)
            # In cylindrical coordinates, φ coordinate derivative not present here.  Use R = r.
            # However, for Boozer transform we need only the Jacobian factor w = R^2 * ∂(λ)/∂θ * 2π?  See theory.
            # Here we compute the covariant basis vectors and the Jacobian determinant
            # e_theta = (dth_r, 0, dth_z)
            # e_zeta = (dze_r, r, dze_z)
            # The Jacobian J = r * (dth_r * dze_z - dze_r * dth_z)
            J = r * (dth_r * dze_z - dze_r * dth_z)
            # Contravariant metric component g^{uv}
            # Boozer condition: bsubu = I + (∂χ/∂θ), bsubv = G + (∂χ/∂ζ)
            # Compute u and v derivatives of Boozer angles: ∂vartheta/∂θ and ∂vphi/∂ζ
            # Solve linear system to enforce B · ∇vartheta = 0 and B · ∇vphi = 1/R^2.
            # For efficiency we solve per grid point.  Use vectorised operations.
            # Compute elements of the matrix and RHS according to the original algorithm.
            # See booz_xform theory for details.
            # For each grid point, solve 2x2 system:
            # [ bsubu   bsubv ] [ u_th  ] = [ 0   ]
            # [ bsubu   bsubv ] [ v_ze  ]   [ Bm ]
            # Additional relations for derivatives cross etc.  Use the original C++ code as guidance.
            #
            # We vectorise by solving for all grid points simultaneously.  Because the matrix is the same in both rows
            # (bsubu and bsubv), the solution simplifies.
            # Let det = bsubu * bsubu + bsubv * bsubv
            det = bsubu * bsubu + bsubv * bsubv
            # Avoid division by zero by adding a small epsilon; this also prevents NaNs when B=0.
            eps = 1e-14
            det = det + eps
            # Solve for derivatives of Boozer angles
            u_th = ( -bsubv * Bm ) / det
            u_ze = (  bsubu * Bm ) / det
            v_th = bsubv / det
            v_ze = -bsubu / det
            # Integrate derivatives to obtain Boozer angles
            # Use cumulative trapezoidal integration over theta and zeta.  Since the grid is uniform,
            # we can approximate the integral by summing derivatives times grid spacing.
            # First integrate over theta (axis 1) then zeta (axis 0).
            # Compute cumulative sums for each variable separately.  JAX's cumsum yields an array of same shape.
            dth = 2.0 * jnp.pi / self._ntheta
            dze = 2.0 * jnp.pi / self._nzeta
            vartheta = jnp.cumsum(u_th * dth, axis=1)
            vphi = jnp.cumsum(u_ze * dze, axis=0)
            # Remove mean to avoid numerical drift
            vartheta = vartheta - jnp.mean(vartheta, axis=(0,1))
            vphi = vphi - jnp.mean(vphi, axis=(0,1))
            # Build complex exponentials for Boozer harmonics
            exp_i = jnp.exp(1j * (jnp.outer(self.xm_b, vartheta.flatten()) - jnp.outer(self.xn_b // self.nfp, vphi.flatten())))
            # Multiply by weights and integrate over the grid to obtain Fourier coefficients
            # The integration measure includes 1/(4π^2) as per the definition.
            weight = J / ( (2.0 * jnp.pi)**2 )
            # Reshape weight to flatten grid
            wvec = weight.flatten()
            # Compute harmonics by dotting with exponential; real and imaginary parts give cos and sin coefficients.
            coeffs = jnp.dot(exp_i, wvec)
            bmnc_b_s = jnp.real(coeffs)
            bmns_b_s = -jnp.imag(coeffs)
            # For R and Z Boozer harmonics, reuse the same exponentials but with r and z values
            coeffs_r = jnp.dot(exp_i, r.flatten()) * (1.0 / (2.0 * jnp.pi)**2)
            coeffs_z = jnp.dot(exp_i, z.flatten()) * (1.0 / (2.0 * jnp.pi)**2)
            rmnc_b_s = jnp.real(coeffs_r)
            rmns_b_s = -jnp.imag(coeffs_r)
            zmns_b_s = jnp.imag(coeffs_z)
            zmnc_b_s = jnp.real(coeffs_z)
            # λ Boozer harmonics produce Boozer angle χ, but we do not expose them directly.
            # Compute Jacobian gmn harmonics
            gtemp = jnp.dot(exp_i, (J * Bm).flatten()) * (1.0 / (2.0 * jnp.pi)**2)
            gmnc_b_s = jnp.real(gtemp)
            gmns_b_s = -jnp.imag(gtemp)
            # Normal derivatives numnc and numns: derivative of B magnitude
            coeffs_nu = jnp.dot(exp_i, (Bm).flatten()) * (1.0 / (2.0 * jnp.pi)**2)
            numnc_b_s = jnp.real(coeffs_nu)
            numns_b_s = -jnp.imag(coeffs_nu)
            if asym_flag:
                return (bmnc_b_s, bmns_b_s, rmnc_b_s, rmns_b_s,
                        zmnc_b_s, zmns_b_s, numnc_b_s, numns_b_s,
                        gmnc_b_s, gmns_b_s)
            else:
                return (bmnc_b_s, None, rmnc_b_s, None,
                        None, zmns_b_s, None, numns_b_s,
                        gmnc_b_s, None)

        # JIT compile the surface solver once
        jit_surface_solve = jax.jit(surface_solve, static_argnums=(12,))
        # Loop over requested surfaces
        for js_b, js in enumerate(self.compute_surfs):
            # Extract VMEC columns for this surface
            rmnc_j = self.rmnc[:, js]
            rmns_j = self.rmns[:, js] if self.asym else None
            zmnc_j = self.zmnc[:, js] if self.asym else None
            zmns_j = self.zmns[:, js]
            lmnc_j = self.lmnc[:, js] if self.asym else None
            lmns_j = self.lmns[:, js]
            bmnc_j = self.bmnc[:, js]
            bmns_j = self.bmns[:, js] if self.asym else None
            bsubumnc_j = self.bsubumnc[:, js]
            bsubumns_j = self.bsubumns[:, js] if self.asym else None
            bsubvmnc_j = self.bsubvmnc[:, js]
            bsubvmns_j = self.bsubvmns[:, js] if self.asym else None
            # Boozer I and G for this surface come from m=0,n=0 entry
            Boozer_I[js_b] = float(bsubumnc_j[0])
            Boozer_G[js_b] = float(bsubvmnc_j[0])
            # Solve surface
            res = jit_surface_solve(rmnc_j, rmns_j, zmnc_j, zmns_j,
                                    lmnc_j, lmns_j, bmnc_j, bmns_j,
                                    bsubumnc_j, bsubumns_j, bsubvmnc_j, bsubvmns_j,
                                    self.asym)
            # Unpack results
            (bmnc_b_s, bmns_b_s, rmnc_b_s, rmns_b_s,
             zmnc_b_s, zmns_b_s, numnc_b_s, numns_b_s,
             gmnc_b_s, gmns_b_s) = res
            # Store into output arrays
            bmnc_b_out[:, js_b] = _np.asarray(bmnc_b_s)
            rmnc_b_out[:, js_b] = _np.asarray(rmnc_b_s)
            if self.asym:
                bmns_b_out[:, js_b] = _np.asarray(bmns_b_s)
                rmns_b_out[:, js_b] = _np.asarray(rmns_b_s)
                zmnc_b_out[:, js_b] = _np.asarray(zmnc_b_s)
                numnc_b_out[:, js_b] = _np.asarray(numnc_b_s)
                gmns_b_out[:, js_b] = _np.asarray(gmns_b_s)
            # Asymmetric or symmetric z and nu harmonics
            if self.asym:
                zmns_b_out[:, js_b] = _np.asarray(zmns_b_s)
                numns_b_out[:, js_b] = _np.asarray(numns_b_s)
            else:
                zmns_b_out[:, js_b] = _np.asarray(zmns_b_s)
                numns_b_out[:, js_b] = _np.asarray(numns_b_s)
            gmnc_b_out[:, js_b] = _np.asarray(gmnc_b_s)
        # Save results on self
        self.bmnc_b = bmnc_b_out
        self.rmnc_b = rmnc_b_out
        self.zmns_b = zmns_b_out
        self.numns_b = numns_b_out
        self.gmnc_b = gmnc_b_out
        if self.asym:
            self.bmns_b = bmns_b_out
            self.rmns_b = rmns_b_out
            self.zmnc_b = zmnc_b_out
            self.numnc_b = numnc_b_out
            self.gmns_b = gmns_b_out
        # Set Boozer radial profiles on selected surfaces
        self.Boozer_I = Boozer_I
        self.Boozer_G = Boozer_G
        # Set s_b for selected surfaces
        self.s_b = _np.asarray(self.s_in)[self.compute_surfs]
        # Results ready
        return None

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