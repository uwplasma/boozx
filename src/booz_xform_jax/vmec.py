"""VMEC input routines for the JAX implementation of ``booz_xform``.

This module contains functions for loading data from VMEC output files
and for initialising a :class:`~booz_xform_jax.core.BoozXform` instance
with that data.  The goal is to mimic the behaviour of the
``booz_xform`` C++ code while providing a Pythonic interface.

The functions defined here operate on instances of
:class:`~booz_xform_jax.core.BoozXform`.  They are not intended to be
called standalone; instead, call the corresponding methods on a
BoozXform object (``init_from_vmec`` and ``read_wout``), which
delegate to these functions.
"""

from __future__ import annotations

import numpy as _np
from typing import Optional

try:
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e

try:
    import netCDF4  # type: ignore
except ImportError:
    netCDF4 = None  # pragma: no cover
try:
    from scipy.io import netcdf_file  # type: ignore
except ImportError:
    netcdf_file = None  # pragma: no cover

def init_from_vmec(self, *args, s_in: Optional[_np.ndarray] = None) -> None:
    """Initialise a :class:`~booz_xform_jax.core.BoozXform` instance with VMEC data.

    This function accepts two calling conventions for compatibility
    with the original C++ ``init_from_vmec`` function:

    1. ``init_from_vmec(ns, iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns,
       bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns[, phip, chi, pres, phi])``
       where ``ns`` is an integer giving the number of radial
       surfaces on the full VMEC grid (including the axis) and the
       remaining arguments are 1D or 2D arrays with shapes matching
       the VMEC documentation.  The zeroth radial entry (the axis) is
       discarded.  Optional arrays ``phip``, ``chi``, ``pres`` and ``phi``
       may appear at the end of the argument list; if present, all
       four must be provided.

    2. ``init_from_vmec(iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns,
       bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns[, phip, chi, pres, phi])``
       omitting the ``ns`` argument.  In this case the length of the
       ``iotas`` array determines the number of surfaces on the full
       grid.  Optional flux arrays may appear at the end as in the
       first calling convention.

    Parameters
    ----------
    self : BoozXform
        The instance to initialise.
    *args : sequence
        Positional arguments following one of the two calling
        conventions described above.
    s_in : ndarray, optional
        Optional array of length ``ns`` giving the values of
        normalized toroidal flux on the full radial grid.  If
        provided, the first entry should correspond to the axis and
        will be discarded.  When not provided, a uniform grid
        between 0 and 1 is used.

    Notes
    -----
    The zeroth radial entry (corresponding to the magnetic axis) is
    ignored in all arrays.  The remaining ``ns - 1`` surfaces
    constitute the half‑grid on which the Boozer transform is
    performed.  All input arrays are defensively copied and
    converted to ``jax.numpy.DeviceArray`` objects of dtype
    ``float64`` for computation; however, the radial coordinate array
    ``s_in`` is stored as a NumPy array so that it can be indexed
    using Python lists in :meth:`~booz_xform_jax.core.BoozXform.run` and
    :func:`~booz_xform_jax.io_utils.read_boozmn`.
    """
    if len(args) == 0:
        raise TypeError("init_from_vmec requires at least one positional argument")
    # Determine if first argument is ns (an integer) or iotas (array)
    first = args[0]
    if isinstance(first, (int, _np.integer)):
        # Signature with ns provided
        if len(args) < 2:
            raise TypeError("init_from_vmec(ns, ...) missing iotas array")
        ns_full = int(first)
        iotas = _np.asarray(args[1])
        arrays = args[2:]
    else:
        # Signature without ns; infer ns from iotas
        iotas = _np.asarray(first)
        ns_full = iotas.shape[0]
        arrays = args[1:]
    # Validate iotas
    if iotas.ndim != 1 or iotas.shape[0] != ns_full:
        raise ValueError("iotas must be a 1D array of length ns")
    if ns_full < 2:
        raise ValueError("ns must be at least 2 surfaces (including the axis)")
    # Validate number of arrays: expect either 12 (no flux) or 16 (with flux)
    if len(arrays) not in (12, 16):
        expected = 14 if isinstance(first, (int, _np.integer)) else 13
        expected_with_flux = expected + 4
        raise TypeError(
            f"init_from_vmec expects {expected} or {expected_with_flux} positional arguments, "
            f"but {len(args)} were given"
        )
    # Determine number of half-grid surfaces and drop the axis
    ns_in = ns_full - 1
    self.ns_in = ns_in
    # Store iota values as JAX array (drop axis)
    self.iota = jnp.asarray(iotas[1:], dtype=jnp.float64)

    # Helper to strip the axis from 2D arrays and convert to JAX.
    # VMEC stores these as (ns_full, mnmax). We want (mnmax, ns_in)
    # internally, with the axis (s=0) removed.
    def strip_axis(arr: _np.ndarray, nrows: int) -> jnp.ndarray:
        if arr.ndim != 2 or arr.shape[0] != nrows:
            raise ValueError(
                f"strip_axis: expected arr.shape[0] == {nrows}, got {arr.shape}"
            )
        # arr: (ns_full, mnmax) → drop s=0 row → (ns_in, mnmax) → transpose
        # to (mnmax, ns_in) for use in core.run
        return jnp.asarray(arr[1:, :].T, dtype=jnp.float64)

    # Unpack mandatory arrays: order rmnc, rmns, zmnc, zmns, lmnc, lmns,
    # bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns.
    (
        rmnc0,
        rmns0,
        zmnc0,
        zmns0,
        lmnc0,
        lmns0,
        bmnc0,
        bmns0,
        bsubumnc0,
        bsubumns0,
        bsubvmnc0,
        bsubvmns0,
    ) = arrays[:12]
    rmnc0 = _np.asarray(rmnc0)
    rmns0 = _np.asarray(rmns0)
    zmnc0 = _np.asarray(zmnc0)
    zmns0 = _np.asarray(zmns0)
    lmnc0 = _np.asarray(lmnc0)
    lmns0 = _np.asarray(lmns0)
    bmnc0 = _np.asarray(bmnc0)
    bmns0 = _np.asarray(bmns0)
    bsubumnc0 = _np.asarray(bsubumnc0)
    bsubumns0 = _np.asarray(bsubumns0)
    bsubvmnc0 = _np.asarray(bsubvmnc0)
    bsubvmns0 = _np.asarray(bsubvmns0)

    # VMEC arrays are (ns_full, mnmax).  The number of modes is the
    # SECOND dimension, not the first.
    self.mnmax = rmnc0.shape[1]

    mnmax = self.mnmax
    asym = self.asym
    xm = _np.asarray(self.xm, dtype=int)  # needs self.xm set from read_wout

    # --- Build full and half s grids like C++ ---
    if s_in is not None:
        # Treat s_in as the full-grid toroidal-flux coordinate, like VMEC
        s_full = _np.asarray(s_in, dtype=float)
        if s_full.shape[0] != ns_full:
            raise ValueError("s_in must have length ns (full grid including axis)")
    else:
        hs = 1.0 / (ns_full - 1.0)
        s_full = hs * _np.arange(ns_full)

    sqrt_s_full = _np.sqrt(s_full)
    sqrt_s_full[0] = 1.0  # avoid div-by-zero; rmnc(s=0)=0 for m>1 anyway

    # Half grid: midpoints between full-grid points
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    sqrt_s_half = _np.sqrt(s_half)

    # Store half-grid s_in (this is what C++ uses internally)
    self.s_in = s_half.astype(float)
    self.ns_in = ns_in

    # --- Radial interpolation for R, Z, lambda on half grid: rmnc, zmns (+ asym parts) ---
    # rmnc0, zmns0, rmns0, zmnc0, lmns0, lmnc0 currently have shape (ns_full, mnmax)
    rmnc_half = _np.empty((mnmax, ns_in), dtype=float)
    zmns_half = _np.empty((mnmax, ns_in), dtype=float)
    lmns_half = _np.empty((mnmax, ns_in), dtype=float)
    if asym:
        rmns_half = _np.empty((mnmax, ns_in), dtype=float)
        zmnc_half = _np.empty((mnmax, ns_in), dtype=float)
        lmnc_half = _np.empty((mnmax, ns_in), dtype=float)
    else:
        rmns_half = zmnc_half = lmnc_half = None

    # Interpolate RMNC, ZMNS and LMNS from full grid (ns_full) to half grid (ns_in).
    # For even m: average adjacent full‑grid points
    # For odd m: interpolate f/√s on the full grid and multiply by √s on the half grid.
    for j in range(mnmax):
        m = int(self.xm[j])  # m values come from read_wout
        if m % 2 == 0:
            rmnc_half[j,:] = 0.5*(rmnc0[:-1,j] + rmnc0[1:,j])
            zmns_half[j,:] = 0.5*(zmns0[:-1,j] + zmns0[1:,j])
            lmns_half[j,:] = 0.5*(lmns0[:-1,j] + lmns0[1:,j])
            if asym:
                rmns_half[j,:] = 0.5*(rmns0[:-1,j] + rmns0[1:,j])
                zmnc_half[j,:] = 0.5*(zmnc0[:-1,j] + zmnc0[1:,j])
                lmnc_half[j,:] = 0.5*(lmnc0[:-1,j] + lmnc0[1:,j])
        else:
            rmnc_half[j,:] = 0.5*((rmnc0[:-1,j]/sqrt_s_full[:-1]) +
                                  (rmnc0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
            zmns_half[j,:] = 0.5*((zmns0[:-1,j]/sqrt_s_full[:-1]) +
                                  (zmns0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
            lmns_half[j,:] = 0.5*((lmns0[:-1,j]/sqrt_s_full[:-1]) +
                                  (lmns0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
            if asym:
                rmns_half[j,:] = 0.5*((rmns0[:-1,j]/sqrt_s_full[:-1]) +
                                      (rmns0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
                zmnc_half[j,:] = 0.5*((zmnc0[:-1,j]/sqrt_s_full[:-1]) +
                                      (zmnc0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
                lmnc_half[j,:] = 0.5*((lmnc0[:-1,j]/sqrt_s_full[:-1]) +
                                      (lmnc0[1:,j]/sqrt_s_full[1:])) * sqrt_s_half
            if m == 1:
                # m=1 extrapolation at the axis
                rmnc_half[j,0] = (1.5*rmnc0[1,j]/sqrt_s_full[1] -
                                  0.5*rmnc0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]
                zmns_half[j,0] = (1.5*zmns0[1,j]/sqrt_s_full[1] -
                                  0.5*zmns0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]
                lmns_half[j,0] = (1.5*lmns0[1,j]/sqrt_s_full[1] -
                                  0.5*lmns0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]
                if asym:
                    rmns_half[j,0] = (1.5*rmns0[1,j]/sqrt_s_full[1] -
                                      0.5*rmns0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]
                    zmnc_half[j,0] = (1.5*zmnc0[1,j]/sqrt_s_full[1] -
                                      0.5*zmnc0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]
                    lmnc_half[j,0] = (1.5*lmnc0[1,j]/sqrt_s_full[1] -
                                      0.5*lmnc0[2,j]/sqrt_s_full[2]) * sqrt_s_half[0]

    # Now store these in the same orientation as the C++ internal rmnc(jmn, js)
    # i.e. (mnmax, ns_in) as JAX arrays:
    self.rmnc = jnp.asarray(rmnc_half, dtype=jnp.float64)
    self.zmns = jnp.asarray(zmns_half, dtype=jnp.float64)
    self.lmns = jnp.asarray(lmns_half, dtype=jnp.float64)
    if asym:
        self.rmns = jnp.asarray(rmns_half, dtype=jnp.float64)
        self.zmnc = jnp.asarray(zmnc_half, dtype=jnp.float64)
        self.lmnc = jnp.asarray(lmnc_half, dtype=jnp.float64)
    else:
        self.rmns = None
        self.zmnc = None
        self.lmnc = None
    # Nyquist arrays: VMEC stores them as (ns_full, mnmax_nyq).
    # Again, we want (mnmax_nyq, ns_in) internally.
    self.mnmax_nyq = bmnc0.shape[1]

    def strip_axis_nyq(arr: _np.ndarray, nrows: int) -> jnp.ndarray:
        if arr.ndim != 2 or arr.shape[0] != nrows:
            raise ValueError(
                f"strip_axis_nyq: expected arr.shape[0] == {nrows}, got {arr.shape}"
            )
        # arr: (ns_full, mnmax_nyq) → drop s=0 row → (ns_in, mnmax_nyq)
        # → transpose to (mnmax_nyq, ns_in).
        return jnp.asarray(arr[1:, :].T, dtype=jnp.float64)

    self.bmnc = strip_axis_nyq(bmnc0, ns_full)
    self.bsubumnc = strip_axis_nyq(bsubumnc0, ns_full)
    self.bsubvmnc = strip_axis_nyq(bsubvmnc0, ns_full)
    if self.asym:
        self.bmns = strip_axis_nyq(bmns0, ns_full)
        self.bsubumns = strip_axis_nyq(bsubumns0, ns_full)
        self.bsubvmns = strip_axis_nyq(bsubvmns0, ns_full)
    else:
        self.bmns = None
        self.bsubumns = None
        self.bsubvmns = None

    # Store Boozer I and G profiles for all half-grid surfaces (as numpy).
    # With our layout (mnmax_nyq, ns_in), the (m=0,n=0) mode is row 0.
    self.Boozer_I_all = _np.asarray(self.bsubumnc[0, :])
    self.Boozer_G_all = _np.asarray(self.bsubvmnc[0, :])
    # Check for flux arrays
    if len(arrays) == 16:
        phip0, chi0, pres0, phi0 = ( _np.asarray(a) for a in arrays[12:16] )
        ns_full = phip0.shape[0]
        ns_in = ns_full - 1
        two_pi = 2.0 * _np.pi

        # Match C++ sizes and scaling
        phip = _np.empty(ns_in + 1, dtype=float)
        chi  = _np.empty(ns_in + 1, dtype=float)
        pres = _np.empty(ns_in + 1, dtype=float)
        phi  = _np.empty(ns_in + 1, dtype=float)

        for j in range(ns_in + 1):
            phip[j] = -phip0[j] / two_pi
            chi[j]  = chi0[j]
            pres[j] = pres0[j]
            phi[j]  = phi0[j]

        # Store flux profiles on the half grid (drop the axis), to be consistent
        # with iota, rmnc, etc., which all live on ns_in surfaces.
        self.phip = jnp.asarray(phip[1:], dtype=jnp.float64)  # length ns_in
        self.chi  = jnp.asarray(chi[1:],  dtype=jnp.float64)
        self.pres = jnp.asarray(pres[1:], dtype=jnp.float64)
        self.phi  = jnp.asarray(phi[1:],  dtype=jnp.float64)

        # Toroidal flux: keep the full-grid last value (outer surface)
        self.toroidal_flux = float(phi[ns_full - 1])
    else:
        self.phip = self.chi = self.pres = self.phi = None
        self.toroidal_flux = 0.0
    # Set default compute_surfs if not already set
    if self.compute_surfs is None:
        self.compute_surfs = list(range(ns_in))
    else:
        # Validate existing indices
        cs = list(self.compute_surfs)
        for idx in cs:
            if idx < 0 or idx >= ns_in:
                raise ValueError(
                    f"compute_surfs has an entry {idx} outside the range [0, {ns_in - 1}]"
                )
        self.compute_surfs = cs
    return None


def read_wout(self, filename: str, flux: bool = False) -> None:
    """Read a VMEC ``wout`` file and populate the internal arrays.

    This routine loads the equilibrium data from a VMEC NetCDF file (the
    file whose name begins with ``wout_``).  The Fourier mode definitions,
    the non‑Nyquist and Nyquist Fourier coefficients, and optional
    flux profiles are read.  Once the data are assembled, this
    function calls :func:`init_from_vmec` on the instance to prepare
    the arrays for the Boozer transformation.

    Parameters
    ----------
    self : BoozXform
        The instance to populate.
    filename : str
        Path to a VMEC wout NetCDF file.
    flux : bool, optional
        If ``True``, the flux profile arrays ``phipf`` (or ``phips``),
        ``chi``, ``pres`` and ``phi`` are read and passed to
        :func:`init_from_vmec`.  When ``False``, these arrays are
        ignored.
    """
    # Open file via netCDF4 or SciPy
    if netCDF4 is not None:
        ds = netCDF4.Dataset(filename, 'r')  # type: ignore
        use_scipy = False
    elif netcdf_file is not None:
        ds = netcdf_file(filename, 'r', mmap=False)  # type: ignore
        use_scipy = True
    else:
        raise RuntimeError("No NetCDF reader available. Install netCDF4 or SciPy.")
    # Read symmetry flag
    # In netCDF4 dimensions are names with double underscores
    lasym_name = 'lasym__logical__'
    if lasym_name in ds.variables:
        lasym = bool(ds.variables[lasym_name][...].item())
    else:
        # Fallback for SciPy file
        lasym = bool(getattr(ds, lasym_name, False))
    self.asym = lasym
    # Read field periodicity
    self.nfp = int(ds.variables['nfp'][...].item())
    # Non‑Nyquist dimension sizes
    self.mpol = int(ds.variables['mpol'][...].item())
    self.ntor = int(ds.variables['ntor'][...].item())
    self.mnmax = int(ds.variables['mnmax'][...].item())
    self.mnmax_nyq = int(ds.variables['mnmax_nyq'][...].item())
    # Read mode number arrays
    self.xm = _np.asarray(ds.variables['xm'][:], dtype=int)
    self.xn = _np.asarray(ds.variables['xn'][:], dtype=int)
    self.xm_nyq = _np.asarray(ds.variables['xm_nyq'][:], dtype=int)
    self.xn_nyq = _np.asarray(ds.variables['xn_nyq'][:], dtype=int)
    self.mpol_nyq = int(self.xm_nyq[-1])
    self.ntor_nyq = int(self.xn_nyq[-1] // self.nfp)
    self.ns_vmec = int(ds.variables['ns'][...].item())
    # Read non-Nyquist Fourier coefficients (shape (mnmax, ns))
    rmnc0 = _np.asarray(ds.variables['rmnc'][:])
    rmns0 = _np.asarray(ds.variables['rmns'][:]) if self.asym else _np.zeros_like(rmnc0)
    zmnc0 = _np.asarray(ds.variables['zmnc'][:]) if self.asym else _np.zeros_like(rmnc0)
    zmns0 = _np.asarray(ds.variables['zmns'][:])
    lmnc0 = _np.asarray(ds.variables['lmnc'][:]) if self.asym else _np.zeros_like(rmnc0)
    lmns0 = _np.asarray(ds.variables['lmns'][:])
    # Read Nyquist Fourier coefficients (shape (mnmax_nyq, ns))
    bmnc0 = _np.asarray(ds.variables['bmnc'][:])
    bmns0 = _np.asarray(ds.variables['bmns'][:]) if self.asym else _np.zeros_like(bmnc0)
    bsubumnc0 = _np.asarray(ds.variables['bsubumnc'][:])
    bsubumns0 = _np.asarray(ds.variables['bsubumns'][:]) if self.asym else _np.zeros_like(bmnc0)
    bsubvmnc0 = _np.asarray(ds.variables['bsubvmnc'][:])
    bsubvmns0 = _np.asarray(ds.variables['bsubvmns'][:]) if self.asym else _np.zeros_like(bmnc0)
    # Determine number of radial surfaces
    ns = rmnc0.shape[0]
    # Initialize variables for flux profiles
    phip0 = chi0 = pres0 = phi0 = None
    if flux:
        # VMEC stores phipf as derivative of toroidal flux, sometimes named phipf or phips
        if 'phipf' in ds.variables:
            phip0 = _np.asarray(ds.variables['phipf'][:])
        elif 'phips' in ds.variables:
            phip0 = _np.asarray(ds.variables['phips'][:])
        if 'chi' in ds.variables:
            chi0 = _np.asarray(ds.variables['chi'][:])
        if 'pres' in ds.variables:
            pres0 = _np.asarray(ds.variables['pres'][:])
        if 'phi' in ds.variables:
            phi0 = _np.asarray(ds.variables['phi'][:])
    # Extract iotas before closing the dataset; shape (ns,)
    iotas = _np.asarray(ds.variables['iotas'][:])
    # Record additional scalar quantities before closing.  The aspect
    # ratio and toroidal flux are stored on the full grid.  We copy
    # them now so that we can safely close the dataset.
    try:
        aspect0 = float(ds.variables['aspect'][...].item())
    except Exception:
        aspect0 = 0.0
    # The toroidal flux is stored in phi0; if phi0 was read above we
    # will extract the last value later; otherwise it remains zero.
    # Extract iotas before closing the dataset; shape (ns,)
    iotas = _np.asarray(ds.variables['iotas'][:])
    # Close dataset
    if use_scipy:
        ds.close()
    else:
        ds.close()
    # Build argument list for init_from_vmec
    args = [ns, iotas]
    # Non-Nyquist arrays
    args.extend([
        rmnc0,
        rmns0,
        zmnc0,
        zmns0,
        lmnc0,
        lmns0,
        bmnc0,
        bmns0,
        bsubumnc0,
        bsubumns0,
        bsubvmnc0,
        bsubvmns0,
    ])
    # Append flux arrays if requested and available
    if flux and phip0 is not None and chi0 is not None and pres0 is not None and phi0 is not None:
        args.extend([phip0, chi0, pres0, phi0])
    # Call init_from_vmec on self
    init_from_vmec(self, *args)
    # Set aspect ratio from stored value
    self.aspect = aspect0

    return None