"""JAX implementation of the Boozer coordinate transformation.

This module provides a :class:`BoozXform` class that mimics the API of the
original C++ code in the ``booz_xform`` package but is implemented purely
in Python using JAX.  The class transforms VMEC Fourier coefficients
into Boozer‐coordinate Fourier coefficients on a specified set of flux
surfaces.  Once created and initialized with VMEC data via
``init_from_vmec``, call ``run`` to perform the coordinate transform.

The implementation follows the algorithm described in the original
``booz_xform`` Fortran and C++ sources.  It makes heavy use of
vectorized operations and JAX's just‑in‑time (JIT) compilation to
achieve good performance.  Users are encouraged to further wrap the
``run`` method with ``jax.jit`` or ``jax.vmap`` if they wish to
differentiate through the Boozer transform or batch over multiple
equilibria.

Note that JAX must be installed on your system for this module to
function.  You can install a CPU‑only build via ``pip install jax jaxlib``.
"""

from __future__ import annotations

import math
import numpy as _np
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e


@dataclass
class BoozXform:
    """Class implementing the Boozer coordinate transformation using JAX.

    Instances of :class:`BoozXform` encapsulate all of the data required
    to convert the spectral representation of a VMEC equilibrium to a
    spectral representation in Boozer coordinates.  After setting up
    mode information (``mpol``, ``ntor``, ``mnmax``, etc.) and then
    populating the VMEC data via :meth:`init_from_vmec`, call
    :meth:`run` to compute the Boozer harmonics on the requested
    surfaces.  The results are stored on the object as ``bmnc_b``,
    ``bmns_b``, ``rmnc_b``, etc. and the mode lists are stored as
    ``xm_b`` and ``xn_b``.

    The API deliberately mirrors the original ``booz_xform`` class so
    that downstream codes such as SIMSOPT can operate on either
    implementation interchangeably.  All arrays are stored as
    :class:`jax.numpy.DeviceArray` objects internally and converted to
    ``numpy.ndarray`` only when necessary for compatibility with
    non‑JAX code.

    Parameters
    ----------
    nfp : int
        The number of field periods of the equilibrium.  Must be at
        least one.
    asym : bool, optional
        If ``True`` the configuration is not stellarator symmetric and
        sine harmonics must be retained.  If ``False`` a
        stellarator‐symmetric assumption is made and certain arrays are
        ignored.
    verbose : int, optional
        Level of diagnostic output printed to standard output.  A value
        of ``0`` suppresses all output, ``1`` prints high‑level
        progress information and ``2`` prints detailed diagnostic
        information.  Note that most printing occurs in Python
        context, not inside jitted functions.
    """

    nfp: int = 1
    asym: bool = False
    verbose: int = 0

    # Mode information for VMEC input
    mpol: int = 0
    ntor: int = 0
    mnmax: int = 0
    xm: Optional[_np.ndarray] = None
    xn: Optional[_np.ndarray] = None

    mpol_nyq: int = 0
    ntor_nyq: int = 0
    mnmax_nyq: int = 0
    xm_nyq: Optional[_np.ndarray] = None
    xn_nyq: Optional[_np.ndarray] = None

    # VMEC data (non‑Nyquist and Nyquist) – will be converted to jax arrays
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

    # List of radial surfaces to compute Boozer data on.  These
    # correspond to half‐grid surfaces and are 0‑based indices into
    # the radial dimension of the VMEC arrays.
    compute_surfs: Optional[Sequence[int]] = None

    # Boozer resolution (output resolution).  These correspond to the
    # m and n truncations of the Boozer harmonics.
    mboz: Optional[int] = None
    nboz: Optional[int] = None

    # Output arrays: will be allocated in run()
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

    # Profiles of I and G for each computed surface
    Boozer_I: Optional[_np.ndarray] = None
    Boozer_G: Optional[_np.ndarray] = None

    # Profiles of I and G on all input surfaces (ns_in).
    Boozer_I_all: Optional[_np.ndarray] = None
    Boozer_G_all: Optional[_np.ndarray] = None

    # Additional VMEC profile data that may be present in boozmn files.
    phi: Optional[jnp.ndarray] = None
    chi: Optional[jnp.ndarray] = None
    phip: Optional[jnp.ndarray] = None
    pres: Optional[jnp.ndarray] = None

    # Optional meta‐data from VMEC equilibrium for output files.
    aspect: float = 0.0
    toroidal_flux: float = 0.0

    # Raw input surfaces (for mapping between s and indices)
    s_in: Optional[jnp.ndarray] = None
    s_b: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if self.nfp < 1:
            raise ValueError("nfp must be at least 1")

    # ------------------------------------------------------------------
    def register_surfaces(self, surfaces: Iterable) -> None:
        """Register a collection of radial surfaces on which to compute Boozer data.

        This convenience method provides a more user‑friendly way to
        specify the ``compute_surfs`` list.  The input may be a sequence
        of integer indices, a sequence of float values in the range
        [0,1] representing normalized toroidal flux, or a mixture of
        both.  When float values are supplied, the closest available
        half‑grid surface in ``self.s_in`` is selected.  Duplicate
        surfaces are removed.  The resulting list is stored in
        ``self.compute_surfs``.

        Parameters
        ----------
        surfaces : Iterable
            A sequence of surfaces to register.  Elements may be
            integers (0‑based indices into the half‑grid) or floats in
            the range [0,1] corresponding to normalized toroidal flux
            values.
        """
        if self.s_in is None:
            raise RuntimeError("init_from_vmec must be called before register_surfaces")
        new_surfs: List[int] = []
        for s in surfaces:
            if isinstance(s, (int, _np.integer)):
                idx = int(s)
                if idx < 0 or idx >= self.iota.shape[0]:
                    raise ValueError(f"Surface index {idx} out of range")
                new_surfs.append(idx)
            else:
                # Treat as float specifying normalized toroidal flux
                s_float = float(s)
                if s_float < 0.0 or s_float > 1.0:
                    raise ValueError("Surface values must lie in [0,1]")
                # Find nearest half‑grid surface
                # Compute absolute difference between s_in and s_float
                diffs = _np.abs(_np.asarray(self.s_in) - s_float)
                idx = int(_np.argmin(diffs))
                new_surfs.append(idx)
        # Remove duplicates and sort
        new_surfs = sorted(set(new_surfs))
        # Store compute_surfs
        self.compute_surfs = new_surfs

    # ------------------------------------------------------------------
    def register(self, surfaces: Iterable) -> None:
        """Alias for :meth:`register_surfaces` for backward compatibility.

        The original ``booz_xform`` C++/Python API uses a method named
        ``register`` to add radial surfaces on which the Boozer transform
        should be computed.  This method simply forwards its arguments
        to :meth:`register_surfaces`.

        Parameters
        ----------
        surfaces : Iterable
            Sequence of surfaces specified either as integer half–grid
            indices or floats in the range [0,1] representing
            normalized toroidal flux values.
        """
        self.register_surfaces(surfaces)

    # ------------------------------------------------------------------
    # VMEC initialization
    def init_from_vmec(self, *args, s_in: Optional[_np.ndarray] = None) -> None:
        """Load Fourier data from VMEC into this instance.

        This method accepts two calling conventions for compatibility with
        the original ``booz_xform`` package:

        1. ``init_from_vmec(ns, iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns, bmnc,
           bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns)``
           where ``ns`` is an integer giving the number of radial
           surfaces on the full VMEC grid and the remaining arguments are
           1D or 2D arrays with the shapes described in the booz_xform
           documentation.  The zeroth radial entry is discarded.

        2. ``init_from_vmec(iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns,
           bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns)``
           omitting the ``ns`` argument.  In this case the length of the
           ``iotas`` array determines the number of surfaces on the full
           grid.

        Parameters
        ----------
        *args : sequence
            Positional arguments following one of the two calling
            conventions described above.  The first argument may be
            either ``ns`` (an integer) or ``iotas`` (a 1D array of
            length ``ns``).  The subsequent arrays must be provided in
            the same order as in the original C++ ``init_from_vmec``
            function.  All arrays must be 1D or 2D ``numpy`` arrays
            containing real numbers.
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
        ``float64``.  Once called, the internal state of this instance
        cannot be reinitialised; to change the VMEC data, construct a
        new :class:`BoozXform` object.
        """

        if len(args) == 0:
            raise TypeError("init_from_vmec requires at least one positional argument")

        # Determine if the first argument is ns (an integer) or iotas (an array)
        first = args[0]
        if isinstance(first, (int, _np.integer)):
            # Signature: ns, iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns, bmnc, bmns,
            #            bsubumnc, bsubumns, bsubvmnc, bsubvmns
            if len(args) < 2:
                raise TypeError("init_from_vmec(ns, ...) missing iotas array")
            ns_full = int(first)
            iotas = _np.asarray(args[1])
            arrays = args[2:]
            # In this calling convention there should be 12 arrays
            if len(arrays) != 12:
                raise TypeError(
                    "init_from_vmec(ns, iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns, "
                    "bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns) expects 14 positional arguments"
                )
        else:
            # Signature: iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns, bmnc, bmns,
            #            bsubumnc, bsubumns, bsubvmnc, bsubvmns
            iotas = _np.asarray(first)
            arrays = args[1:]
            ns_full = iotas.shape[0]
            if len(arrays) != 12:
                raise TypeError(
                    "init_from_vmec(iotas, rmnc, rmns, zmnc, zmns, lmnc, lmns, "
                    "bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns) expects 13 positional arguments"
                )

        if iotas.ndim != 1 or iotas.shape[0] != ns_full:
            raise ValueError("iotas must be a 1D array of length ns")
        if ns_full < 2:
            raise ValueError("ns must be at least 2 surfaces (including the axis)")

        # Determine number of half‑grid surfaces (ns_in) and drop axis
        ns_in = ns_full - 1
        self.ns_in = ns_in

        # Store iota values, discarding the axis entry
        self.iota = jnp.asarray(iotas[1:], dtype=jnp.float64)

        # Determine s_in: if provided, validate length; else generate uniform grid
        if s_in is not None:
            s_in = _np.asarray(s_in)
            if s_in.shape[0] != ns_full:
                raise ValueError("s_in must have the same length as iotas/ns")
            self.s_in = jnp.asarray(s_in[1:], dtype=jnp.float64)
        else:
            # Generate a default uniform grid on [0, 1] of length ns_full
            self.s_in = jnp.linspace(0.0, 1.0, ns_full, dtype=jnp.float64)[1:]

        # Helper for copying and discarding the first radial entry of 2D arrays
        def strip_axis(arr: _np.ndarray, expect_mnmax: bool = True) -> jnp.ndarray:
            if arr.ndim != 2 or arr.shape[1] != ns_full:
                raise ValueError("Input array has wrong shape: expected (mnmax, ns)")
            return jnp.asarray(arr[:, 1:], dtype=jnp.float64)

        # Unpack arrays: order is rmnc, rmns, zmnc, zmns, lmnc, lmns, bmnc, bmns,
        # bsubumnc, bsubumns, bsubvmnc, bsubvmns
        rmnc0, rmns0, zmnc0, zmns0, lmnc0, lmns0, bmnc0, bmns0, bsubumnc0, bsubumns0, bsubvmnc0, bsubvmns0 = arrays

        # Copy non‑Nyquist quantities (shape mnmax x ns)
        self.rmnc = strip_axis(_np.asarray(rmnc0))
        self.zmns = strip_axis(_np.asarray(zmns0))
        self.lmns = strip_axis(_np.asarray(lmns0))
        if self.asym:
            self.rmns = strip_axis(_np.asarray(rmns0))
            self.zmnc = strip_axis(_np.asarray(zmnc0))
            self.lmnc = strip_axis(_np.asarray(lmnc0))
        else:
            self.rmns = None
            self.zmnc = None
            self.lmnc = None

        # Copy Nyquist quantities (shape mnmax_nyq x ns)
        def strip_axis_nyq(arr: _np.ndarray) -> jnp.ndarray:
            if arr.ndim != 2 or arr.shape[1] != ns_full:
                raise ValueError(
                    "Input Nyquist array has wrong shape: expected (mnmax_nyq, ns)"
                )
            return jnp.asarray(arr[:, 1:], dtype=jnp.float64)

        self.bmnc = strip_axis_nyq(_np.asarray(bmnc0))
        self.bsubumnc = strip_axis_nyq(_np.asarray(bsubumnc0))
        self.bsubvmnc = strip_axis_nyq(_np.asarray(bsubvmnc0))
        if self.asym:
            self.bmns = strip_axis_nyq(_np.asarray(bmns0))
            self.bsubumns = strip_axis_nyq(_np.asarray(bsubumns0))
            self.bsubvmns = strip_axis_nyq(_np.asarray(bsubvmns0))
        else:
            self.bmns = None
            self.bsubumns = None
            self.bsubvmns = None

        # Compute Boozer I and G profiles for all input surfaces.  These
        # correspond to the (m=0,n=0) Nyquist mode of bsubumnc and bsubvmnc.
        # Both arrays have been stripped of the axis entry.
        # The zeroth row of bsubumnc and bsubvmnc corresponds to m=0, n=0.
        # Use numpy arrays here since these values are small vectors and
        # writing to netCDF requires numpy data.
        self.Boozer_I_all = _np.asarray(self.bsubumnc[0, :])
        self.Boozer_G_all = _np.asarray(self.bsubvmnc[0, :])

        # Check for extra input arrays (phi, chi, pres, phip).  These are
        # defined on the full VMEC radial grid.  If present, store them
        # after discarding the axis entry.  Remaining arrays in
        # ``arrays`` beyond the required 12 will be interpreted as
        # phip0, chi0, pres0, phi0, respectively.
        # At most four additional arrays are accepted.
        extra_count = len(arrays) - 12
        if extra_count >= 4:
            phip0, chi0, pres0, phi0 = arrays[12:16]
            # Note: the original C++ code stores these arrays on the
            # full grid (ns elements).  We drop the axis entry for
            # consistency with other radial profiles.
            self.phip = jnp.asarray(_np.asarray(phip0)[1:], dtype=jnp.float64)
            self.chi = jnp.asarray(_np.asarray(chi0)[1:], dtype=jnp.float64)
            self.pres = jnp.asarray(_np.asarray(pres0)[1:], dtype=jnp.float64)
            self.phi = jnp.asarray(_np.asarray(phi0)[1:], dtype=jnp.float64)
        # Set default compute_surfs if not specified
        if self.compute_surfs is None:
            self.compute_surfs = list(range(ns_in))
        else:
            # Validate existing compute_surfs indices
            cs = list(self.compute_surfs)
            for idx in cs:
                if idx < 0 or idx >= ns_in:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside the range [0, {ns_in - 1}]"
                    )
            self.compute_surfs = cs

        # No further processing until run() is called

    # ------------------------------------------------------------------
    def read_wout(self, filename: str, flux: bool = False) -> None:
        """Read VMEC wout file and populate the internal arrays.

        This method loads the equilibrium data from a VMEC ``wout_*.nc`` file.
        It reads the Fourier mode definitions, the non‑Nyquist and Nyquist
        Fourier coefficients, and optional flux profiles.  Once the data
        are read, :meth:`init_from_vmec` is called to prepare the arrays
        for the Boozer transformation.  The aspect ratio and field
        periodicity are stored, and the lists of Fourier mode numbers
        are saved on the instance.

        Parameters
        ----------
        filename : str
            Path to a VMEC wout NetCDF file.
        flux : bool, optional
            If ``True``, the flux profile arrays ``phipf``, ``chi``,
            ``pres`` and ``phi`` are read and stored.  When ``False``
            these arrays are ignored.
        """
        # Use SciPy if netCDF4 is unavailable or as fallback
        try:
            import netCDF4 as nc  # type: ignore
            ds = nc.Dataset(filename, 'r')
            use_scipy = False
        except Exception:
            from scipy.io import netcdf_file  # type: ignore
            ds = netcdf_file(filename, 'r')
            use_scipy = True
        # Read symmetry flag
        if use_scipy:
            asym_val = ds.variables['lasym__logical__'].data if 'lasym__logical__' in ds.variables else ds.variables['lasym__logical__'][:]
            self.asym = bool(int(asym_val))
            # Scalars
            ns = int(ds.variables['ns'].data if 'ns' in ds.variables else ds.variables['ns'][:])
            self.nfp = int(ds.variables['nfp'].data if 'nfp' in ds.variables else ds.variables['nfp'][:])
            self.mpol = int(ds.variables['mpol'].data if 'mpol' in ds.variables else ds.variables['mpol'][:])
            self.ntor = int(ds.variables['ntor'].data if 'ntor' in ds.variables else ds.variables['ntor'][:])
            self.mnmax = int(ds.variables['mnmax'].data if 'mnmax' in ds.variables else ds.variables['mnmax'][:])
            self.mnmax_nyq = int(ds.variables['mnmax_nyq'].data if 'mnmax_nyq' in ds.variables else ds.variables['mnmax_nyq'][:])
            # Aspect ratio if available
            if 'aspect' in ds.variables:
                self.aspect = float(ds.variables['aspect'].data)
            # Mode lists
            xm = _np.asarray(ds.variables['xm'][:], dtype=int)
            xn = _np.asarray(ds.variables['xn'][:], dtype=int)
            xm_nyq = _np.asarray(ds.variables['xm_nyq'][:], dtype=int)
            xn_nyq = _np.asarray(ds.variables['xn_nyq'][:], dtype=int)
        else:
            asym_val = ds.variables['lasym__logical__'][...].item() if 'lasym__logical__' in ds.variables else ds.variables['lasym__logical__'][:]
            self.asym = bool(int(asym_val))
            ns = int(ds.variables['ns'][...].item())
            self.nfp = int(ds.variables['nfp'][...].item())
            self.mpol = int(ds.variables['mpol'][...].item())
            self.ntor = int(ds.variables['ntor'][...].item())
            self.mnmax = int(ds.variables['mnmax'][...].item())
            self.mnmax_nyq = int(ds.variables['mnmax_nyq'][...].item())
            # Aspect ratio if present
            if 'aspect' in ds.variables:
                self.aspect = float(ds.variables['aspect'][...].item())
            # Mode lists
            xm = _np.asarray(ds.variables['xm'][:], dtype=int)
            xn = _np.asarray(ds.variables['xn'][:], dtype=int)
            xm_nyq = _np.asarray(ds.variables['xm_nyq'][:], dtype=int)
            xn_nyq = _np.asarray(ds.variables['xn_nyq'][:], dtype=int)
        # Store mode lists and compute Nyquist truncations
        self.xm = xm
        self.xn = xn
        self.xm_nyq = xm_nyq
        self.xn_nyq = xn_nyq
        # Determine mpol_nyq and ntor_nyq from last elements
        if xm_nyq.size > 0:
            self.mpol_nyq = int(xm_nyq[-1])
            self.ntor_nyq = int(xn_nyq[-1] // self.nfp)
        else:
            self.mpol_nyq = 0
            self.ntor_nyq = 0
        # Read iota
        iotas = _np.asarray(ds.variables['iotas'][:]) if use_scipy else _np.asarray(ds.variables['iotas'][:])
        # Non‑Nyquist arrays: shapes (ns, mnmax) in file.  Transpose to (mnmax, ns).
        def read_and_transpose(name):
            arr = _np.asarray(ds.variables[name][:])
            return arr.T
        rmnc0 = read_and_transpose('rmnc')
        zmns0 = read_and_transpose('zmns')
        lmns0 = read_and_transpose('lmns')
        # For asymmetric case, these arrays may not exist
        rmns0 = None
        zmnc0 = None
        lmnc0 = None
        if self.asym:
            if 'rmns' in ds.variables:
                rmns0 = read_and_transpose('rmns')
            else:
                rmns0 = _np.zeros_like(rmnc0)
            if 'zmnc' in ds.variables:
                zmnc0 = read_and_transpose('zmnc')
            else:
                zmnc0 = _np.zeros_like(rmnc0)
            if 'lmnc' in ds.variables:
                lmnc0 = read_and_transpose('lmnc')
            else:
                lmnc0 = _np.zeros_like(lmns0)
        # Nyquist arrays: shapes (ns, mnmax_nyq).  Transpose to (mnmax_nyq, ns).
        def read_and_transpose_nyq(name):
            arr = _np.asarray(ds.variables[name][:])
            return arr.T
        bmnc0 = read_and_transpose_nyq('bmnc')
        bsubumnc0 = read_and_transpose_nyq('bsubumnc')
        bsubvmnc0 = read_and_transpose_nyq('bsubvmnc')
        bmns0 = None
        bsubumns0 = None
        bsubvmns0 = None
        if self.asym:
            if 'bmns' in ds.variables:
                bmns0 = read_and_transpose_nyq('bmns')
            else:
                bmns0 = _np.zeros_like(bmnc0)
            if 'bsubumns' in ds.variables:
                bsubumns0 = read_and_transpose_nyq('bsubumns')
            else:
                bsubumns0 = _np.zeros_like(bmnc0)
            if 'bsubvmns' in ds.variables:
                bsubvmns0 = read_and_transpose_nyq('bsubvmns')
            else:
                bsubvmns0 = _np.zeros_like(bmnc0)
        # Optional flux profiles
        phip0 = None
        chi0 = None
        pres0 = None
        phi0 = None
        if flux:
            # VMEC stores phipf as derivative of toroidal flux, chi as poloidal flux grid,
            # pres as pressure, phi as toroidal flux.  These arrays have length ns.
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
        # Close dataset
        if use_scipy:
            ds.close()
        else:
            ds.close()
        # Call init_from_vmec with or without flux profiles
        if flux and phip0 is not None and chi0 is not None and pres0 is not None and phi0 is not None:
            self.init_from_vmec(
                ns,
                iotas,
                rmnc0,
                rmns0 if rmns0 is not None else _np.zeros_like(rmnc0),
                zmnc0 if zmnc0 is not None else _np.zeros_like(zmns0),
                zmns0,
                lmnc0 if lmnc0 is not None else _np.zeros_like(lmns0),
                lmns0,
                bmnc0,
                bmns0 if bmns0 is not None else _np.zeros_like(bmnc0),
                bsubumnc0,
                bsubumns0 if bsubumns0 is not None else _np.zeros_like(bmnc0),
                bsubvmnc0,
                bsubvmns0 if bsubvmns0 is not None else _np.zeros_like(bmnc0),
                phip0,
                chi0,
                pres0,
                phi0,
            )
        else:
            # supply zeros for missing arrays
            self.init_from_vmec(
                ns,
                iotas,
                rmnc0,
                rmns0 if rmns0 is not None else _np.zeros_like(rmnc0),
                zmnc0 if zmnc0 is not None else _np.zeros_like(zmns0),
                zmns0,
                lmnc0 if lmnc0 is not None else _np.zeros_like(lmns0),
                lmns0,
                bmnc0,
                bmns0 if bmns0 is not None else _np.zeros_like(bmnc0),
                bsubumnc0,
                bsubumns0 if bsubumns0 is not None else _np.zeros_like(bmnc0),
                bsubvmnc0,
                bsubvmns0 if bsubvmns0 is not None else _np.zeros_like(bmnc0),
            )
        # Store toroidal_flux if phi was read
        if flux and phi0 is not None:
            # The last value of phi corresponds to the total toroidal flux (not divided by 2π)
            # in the VMEC file.  Use this as toroidal_flux.
            try:
                self.toroidal_flux = float(phi0[-1])
            except Exception:
                self.toroidal_flux = 0.0
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
        assert self.mboz is not None
        assert self.nboz is not None
        m_list: List[int] = []
        n_list: List[int] = []
        for m in range(self.mboz):
            nmin = 0 if m == 0 else -self.nboz
            for n in range(nmin, self.nboz + 1):
                m_list.append(m)
                n_list.append(n * self.nfp)
        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)

    # ------------------------------------------------------------------
    def _setup_grids(self) -> None:
        """Generate the real‑space grids and trigonometric tables.

        The number of points in the poloidal and toroidal directions are
        determined from ``mboz`` and ``nboz`` following the formulas in
        the original code.  Sine and cosine tables for both the
        non‑Nyquist and Nyquist mode ranges are computed and stored
        as JAX arrays.  These tables are used to accelerate the
        evaluation of Fourier sums.
        """
        # Determine resolution of the real‑space grid
        # Poloidal grid: ntheta = 2*(2*mboz+1)
        ntheta = 2 * (2 * self.mboz + 1)
        # Toroidal grid: nzeta = 2*(2*nboz+1), except when nboz==0
        if self.nboz == 0:
            nzeta = 1
        else:
            nzeta = 2 * (2 * self.nboz + 1)

        # Radial indexing: nu2_b = ntheta//2 + 1; nu3_b depends on asym
        nu2_b = ntheta // 2 + 1
        if self.asym:
            nu3_b = ntheta
        else:
            nu3_b = nu2_b
        # Save for later use
        self._ntheta = ntheta
        self._nzeta = nzeta
        self._nu2_b = nu2_b
        self._nu3_b = nu3_b
        # Spacing in theta
        if self.asym:
            d_theta = 2.0 * math.pi / nu3_b
        else:
            d_theta = 2.0 * math.pi / (2.0 * (nu3_b - 1))
        # Spacing in zeta
        d_zeta = 2.0 * math.pi / (self.nfp * nzeta)
        self._d_theta = d_theta
        self._d_zeta = d_zeta
        # Flattened grid of theta and zeta
        # theta_grid has length nu3_b * nzeta; likewise zeta_grid
        thetas = jnp.arange(nu3_b) * d_theta
        zetas = jnp.arange(nzeta) * d_zeta
        # Create 2D meshgrid and flatten
        theta_grid = jnp.repeat(thetas, nzeta)
        zeta_grid = jnp.tile(zetas, nu3_b)
        self._theta_grid = theta_grid
        self._zeta_grid = zeta_grid
        n_theta_zeta = nu3_b * nzeta
        self._n_theta_zeta = n_theta_zeta

        # Precompute sine and cosine tables for the non‑Nyquist modes
        # Poloidal harmonics run from 0..mpol-1
        m_range = jnp.arange(self.mpol)
        # Toroidal harmonics run from 0..ntor
        n_range = jnp.arange(self.ntor + 1)
        # The following shapes are (n_theta_zeta, mpol) or (n_theta_zeta, ntor+1)
        self._cosm = jnp.cos(theta_grid[:, None] * m_range[None, :])
        self._sinm = jnp.sin(theta_grid[:, None] * m_range[None, :])
        self._cosn = jnp.cos(zeta_grid[:, None] * n_range[None, :])
        self._sinn = jnp.sin(zeta_grid[:, None] * n_range[None, :])

        # Precompute sine and cosine tables for the Nyquist modes
        m_range_nyq = jnp.arange(self.mpol_nyq + 1)
        n_range_nyq = jnp.arange(self.ntor_nyq + 1)
        self._cosm_nyq = jnp.cos(theta_grid[:, None] * m_range_nyq[None, :])
        self._sinm_nyq = jnp.sin(theta_grid[:, None] * m_range_nyq[None, :])
        self._cosn_nyq = jnp.cos(zeta_grid[:, None] * n_range_nyq[None, :])
        self._sinn_nyq = jnp.sin(zeta_grid[:, None] * n_range_nyq[None, :])

    # ------------------------------------------------------------------
    def run(
        self,
        mboz: Optional[int] = None,
        nboz: Optional[int] = None,
        compute_surfs: Optional[Sequence[int]] = None,
    ) -> None:
        """Perform the Boozer coordinate transformation on the selected surfaces.

        This method carries out the full transformation from the VMEC Fourier
        representation to the Boozer representation.  If ``mboz`` or
        ``nboz`` are provided they override the instance attributes for
        the output resolution.  Likewise ``compute_surfs`` can be used
        to override the list of surfaces specified in
        :meth:`init_from_vmec`.  Upon successful completion the
        Fourier coefficients in Boozer coordinates are stored on the
        instance in arrays such as ``bmnc_b`` and ``rmnc_b``.  The
        lists of corresponding mode numbers are stored in ``xm_b`` and
        ``xn_b``.

        Parameters
        ----------
        mboz, nboz : int, optional
            Override the poloidal and toroidal truncations of the Boozer
            harmonics.  By default, the transform uses the same
            truncations as the VMEC input (``self.mpol``, ``self.ntor``).
        compute_surfs : sequence of int, optional
            Override the list of radial indices on which to compute the
            Boozer transform.  The indices must be 0‑based and refer
            to positions in the half‐grid (i.e. they range from 0 to
            ``ns_in-1`` where ``ns_in`` is ``len(self.iota)``).  If
            provided, this argument supersedes the ``compute_surfs``
            specified via :meth:`init_from_vmec`.
        """
        if self.iota is None:
            raise RuntimeError("init_from_vmec must be called before run")

        # Choose resolution for Boozer harmonics
        if mboz is not None:
            self.mboz = mboz
        elif self.mboz is None:
            # default to twice the VMEC poloidal truncation minus one
            # This matches typical choices used in booz_xform
            self.mboz = self.mpol
        if nboz is not None:
            self.nboz = nboz
        elif self.nboz is None:
            self.nboz = self.ntor

        # Override compute_surfs if provided
        if compute_surfs is not None:
            # Validate indices
            cs = list(compute_surfs)
            for idx in cs:
                if idx < 0 or idx >= self.iota.shape[0]:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside the range [0, {self.iota.shape[0] - 1}]"
                    )
            self.compute_surfs = cs

        # Prepare Boozer mode lists
        self._prepare_mode_lists()
        mnboz = self.xm_b.shape[0]

        # Generate real‑space grids and trigonometric tables
        self._setup_grids()

        # Precompute VMEC mode lists and signs for non‑Nyquist
        xm_arr = jnp.asarray(self.xm, dtype=jnp.int32)
        xn_arr = jnp.asarray(self.xn, dtype=jnp.int32)
        # absolute n index (dividing by nfp) for non‑Nyquist
        abs_n_non = jnp.abs(xn_arr) // self.nfp
        sign_non = jnp.where(xn_arr < 0, -1.0, 1.0)

        # Precompute selection matrices for non‑Nyquist modes
        cos_m_non = self._cosm[:, xm_arr]  # shape (n_theta_zeta, mnmax)
        sin_m_non = self._sinm[:, xm_arr]
        cos_n_non = self._cosn[:, abs_n_non]
        sin_n_non = self._sinn[:, abs_n_non]

        # Precompute selection matrices for Nyquist modes
        xm_nyq_arr = jnp.asarray(self.xm_nyq, dtype=jnp.int32)
        xn_nyq_arr = jnp.asarray(self.xn_nyq, dtype=jnp.int32)
        abs_n_nyq = jnp.abs(xn_nyq_arr) // self.nfp
        sign_nyq = jnp.where(xn_nyq_arr < 0, -1.0, 1.0)

        cos_m_nyq = self._cosm_nyq[:, xm_nyq_arr]
        sin_m_nyq = self._sinm_nyq[:, xm_nyq_arr]
        cos_n_nyq = self._cosn_nyq[:, abs_n_nyq]
        sin_n_nyq = self._sinn_nyq[:, abs_n_nyq]

        # Precompute factors for derivatives (m and n lists)
        m_list_non = xm_arr.astype(jnp.float64)
        n_list_non = xn_arr.astype(jnp.float64)
        m_list_nyq = xm_nyq_arr.astype(jnp.float64)
        n_list_nyq = xn_nyq_arr.astype(jnp.float64)

        # Precompute Boozer mode sign arrays for Fourier synthesis
        xm_b_arr = jnp.asarray(self.xm_b, dtype=jnp.int32)
        xn_b_arr = jnp.asarray(self.xn_b, dtype=jnp.int32)
        abs_n_b = jnp.abs(xn_b_arr) // self.nfp
        sign_b = jnp.where(xn_b_arr < 0, -1.0, 1.0)

        # Factor used in final Fourier synthesis
        if self.asym:
            fourier_factor0 = 2.0 / (self._ntheta * self._nzeta)
        else:
            fourier_factor0 = 2.0 / ((self._nu2_b - 1) * self._nzeta)
        # fourier_factor is length mnboz; scale first element by 0.5
        fourier_factor = jnp.full((mnboz,), fourier_factor0, dtype=jnp.float64)
        fourier_factor = fourier_factor.at[0].set(fourier_factor0 * 0.5)

        # Output arrays initialised to zero
        ns_b = len(self.compute_surfs)
        bmnc_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64)
        bmns_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64) if self.asym else None
        rmnc_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64)
        rmns_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64) if self.asym else None
        zmnc_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64) if self.asym else None
        zmns_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64)
        numnc_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64) if self.asym else None
        numns_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64)
        gmnc_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64)
        gmns_b_out = jnp.zeros((mnboz, ns_b), dtype=jnp.float64) if self.asym else None

        # Boozer I and G profiles for each computed surface
        Boozer_I = _np.zeros(ns_b, dtype=float)
        Boozer_G = _np.zeros(ns_b, dtype=float)

        # Helper function computing Boozer harmonics for one surface
        def surface_solve(
            rmnc_j: jnp.ndarray,
            rmns_j: Optional[jnp.ndarray],
            zmnc_j: Optional[jnp.ndarray],
            zmns_j: jnp.ndarray,
            lmnc_j: Optional[jnp.ndarray],
            lmns_j: jnp.ndarray,
            bmnc_j: jnp.ndarray,
            bmns_j: Optional[jnp.ndarray],
            bsubumnc_j: jnp.ndarray,
            bsubumns_j: Optional[jnp.ndarray],
            bsubvmnc_j: jnp.ndarray,
            bsubvmns_j: Optional[jnp.ndarray],
            iota_j: float,
            Boozer_I_j: float,
            Boozer_G_j: float,
        ) -> tuple:
            """Compute Boozer Fourier coefficients on a single surface.

            Parameters
            ----------
            rmnc_j, rmns_j, zmnc_j, zmns_j, lmnc_j, lmns_j : jnp.ndarray
                The VMEC non‑Nyquist Fourier coefficients on the current surface.
            bmnc_j, bmns_j, bsubumnc_j, bsubumns_j, bsubvmnc_j, bsubvmns_j : jnp.ndarray
                The VMEC Nyquist Fourier coefficients on the current surface.
            iota_j : float
                The rotational transform at this surface.
            Boozer_I_j, Boozer_G_j : float
                The values of the Boozer I and G profiles for this surface.

            Returns
            -------
            tuple
                The computed Boozer Fourier coefficients for this surface.
                The order of returned arrays is
                ``(bmnc_b_s, bmns_b_s, rmnc_b_s, rmns_b_s, zmnc_b_s, zmns_b_s,
                numnc_b_s, numns_b_s, gmnc_b_s, gmns_b_s)``.  When ``self.asym``
                is ``False``, the corresponding sine parts are returned as
                ``None``.
            """
            # Non‑Nyquist contributions: evaluate R, Z, lambda and their derivatives
            # Build tcos and tsin for non‑Nyquist modes
            tcos_non = cos_m_non * cos_n_non + sin_m_non * sin_n_non * sign_non
            tsin_non = sin_m_non * cos_n_non - cos_m_non * sin_n_non * sign_non

            # Real‑space functions on the (theta,zeta) grid
            # r = sum_jmn tcos_non[:,jmn] * rmnc_j[jmn]
            r_vals = jnp.dot(tcos_non, rmnc_j)
            z_vals = jnp.dot(tsin_non, zmns_j)
            lambda_vals = jnp.dot(tsin_non, lmns_j)
            # Derivatives
            d_lambda_d_theta = jnp.dot(tcos_non * m_list_non, lmns_j)
            d_lambda_d_zeta = -jnp.dot(tcos_non * n_list_non, lmns_j)

            if self.asym:
                # Additional terms for asymmetric configurations
                r_vals = r_vals + jnp.dot(tsin_non, rmns_j)
                z_vals = z_vals + jnp.dot(tcos_non, zmnc_j)
                lambda_vals = lambda_vals + jnp.dot(tcos_non, lmnc_j)
                d_lambda_d_theta = d_lambda_d_theta - jnp.dot(tsin_non * m_list_non, lmnc_j)
                d_lambda_d_zeta = d_lambda_d_zeta + jnp.dot(tsin_non * n_list_non, lmnc_j)

            # Nyquist contributions: compute w, dw/dtheta, dw/dzeta, bmod
            tcos_nyq = cos_m_nyq * cos_n_nyq + sin_m_nyq * sin_n_nyq * sign_nyq
            tsin_nyq = sin_m_nyq * cos_n_nyq - cos_m_nyq * sin_n_nyq * sign_nyq

            # Compute wmns and wmnc arrays using vmec bsub arrays
            # When m != 0: wmns = bsubumnc/m; wmnc = -bsubumns/m
            # When m == 0 and n != 0: wmns = -bsubvmnc/n; wmnc = bsubvmns/n
            # When m == 0 and n == 0: Boozer_I and Boozer_G already set
            m_nonzero = m_list_nyq != 0.0
            n_nonzero = n_list_nyq != 0.0
            wmns = jnp.where(
                m_nonzero,
                bsubumnc_j / m_list_nyq,
                jnp.where(n_nonzero, -bsubvmnc_j / n_list_nyq, 0.0),
            )
            if self.asym:
                wmnc = jnp.where(
                    m_nonzero,
                    -bsubumns_j / m_list_nyq,
                    jnp.where(n_nonzero, bsubvmns_j / n_list_nyq, 0.0),
                )
            else:
                wmnc = None

            # Evaluate w and its derivatives
            w_vals = jnp.dot(tsin_nyq, wmns)
            dw_d_theta = jnp.dot(tcos_nyq * m_list_nyq, wmns)
            dw_d_zeta = -jnp.dot(tcos_nyq * n_list_nyq, wmns)
            if self.asym:
                w_vals = w_vals + jnp.dot(tcos_nyq, wmnc)
                dw_d_theta = dw_d_theta - jnp.dot(tsin_nyq * m_list_nyq, wmnc)
                dw_d_zeta = dw_d_zeta + jnp.dot(tsin_nyq * n_list_nyq, wmnc)

            # Magnetic field strength
            bmod = jnp.dot(tcos_nyq, bmnc_j)
            if self.asym:
                bmod = bmod + jnp.dot(tsin_nyq, bmns_j)

            # Compute nu from Eq (10)
            GI = Boozer_G_j + iota_j * Boozer_I_j
            one_over_GI = 1.0 / GI
            nu_vals = one_over_GI * (w_vals - Boozer_I_j * lambda_vals)
            dnu_d_theta = one_over_GI * (dw_d_theta - Boozer_I_j * d_lambda_d_theta)
            dnu_d_zeta = one_over_GI * (dw_d_zeta - Boozer_I_j * d_lambda_d_zeta)

            # Compute Boozer angles
            theta_booz = self._theta_grid + lambda_vals + iota_j * nu_vals
            zeta_booz = self._zeta_grid + nu_vals

            # Jacobian derivative d(Boozer)/d(VMEC) (Eq (12))
            dBdz = (1.0 + d_lambda_d_theta) * (1.0 + dnu_d_zeta) + (iota_j - d_lambda_d_zeta) * dnu_d_theta

            # Boozer Jacobian (G + iota*I)/|B|^2
            boozer_jac = GI / (bmod * bmod)

            # Prepare trigonometric tables for Boozer angles up to m=mboz, n=nboz
            m_range_b = jnp.arange(self.mboz + 1)
            n_range_b = jnp.arange(self.nboz + 1)
            cos_m_b = jnp.cos(theta_booz[:, None] * m_range_b[None, :])
            sin_m_b = jnp.sin(theta_booz[:, None] * m_range_b[None, :])
            cos_n_b = jnp.cos(zeta_booz[:, None] * n_range_b[None, :])
            sin_n_b = jnp.sin(zeta_booz[:, None] * n_range_b[None, :])

            # Construct tcos and tsin for each Boozer harmonic
            cos_m_selected = cos_m_b[:, xm_b_arr]
            sin_m_selected = sin_m_b[:, xm_b_arr]
            cos_n_selected = cos_n_b[:, abs_n_b]
            sin_n_selected = sin_n_b[:, abs_n_b]
            tcos_b = cos_m_selected * cos_n_selected + sin_m_selected * sin_n_selected * sign_b
            tsin_b = sin_m_selected * cos_n_selected - cos_m_selected * sin_n_selected * sign_b

            # Multiply by derivative term and integrate over grid
            tcos_weighted = tcos_b * dBdz[:, None]
            tsin_weighted = tsin_b * dBdz[:, None]

            # Compute Fourier coefficients via matrix multiplication
            bmnc_b_s = (jnp.dot(tcos_weighted.T, bmod) * fourier_factor)
            rmnc_b_s = (jnp.dot(tcos_weighted.T, r_vals) * fourier_factor)
            zmns_b_s = (jnp.dot(tsin_weighted.T, z_vals) * fourier_factor)
            numns_b_s = (jnp.dot(tsin_weighted.T, nu_vals) * fourier_factor)
            gmnc_b_s = (jnp.dot(tcos_weighted.T, boozer_jac) * fourier_factor)

            if self.asym:
                bmns_b_s = (jnp.dot(tsin_weighted.T, bmod) * fourier_factor)
                rmns_b_s = (jnp.dot(tsin_weighted.T, r_vals) * fourier_factor)
                zmnc_b_s = (jnp.dot(tcos_weighted.T, z_vals) * fourier_factor)
                numnc_b_s = (jnp.dot(tcos_weighted.T, nu_vals) * fourier_factor)
                gmns_b_s = (jnp.dot(tsin_weighted.T, boozer_jac) * fourier_factor)
            else:
                bmns_b_s = None
                rmns_b_s = None
                zmnc_b_s = None
                numnc_b_s = None
                gmns_b_s = None

            return (
                bmnc_b_s,
                bmns_b_s,
                rmnc_b_s,
                rmns_b_s,
                zmnc_b_s,
                zmns_b_s,
                numnc_b_s,
                numns_b_s,
                gmnc_b_s,
                gmns_b_s,
            )

        # JIT compile the surface solver.  We fix static arguments via closure.
        jit_surface_solve = jax.jit(surface_solve, static_argnums=())

        # Loop over requested surfaces and accumulate results
        for js_b, js in enumerate(self.compute_surfs):
            # Extract columns for this surface from VMEC data
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

            # Boozer I and G for this surface come from the (m=0,n=0) entry
            Boozer_I_j = float(bsubumnc_j[0])
            Boozer_G_j = float(bsubvmnc_j[0])
            Boozer_I[js_b] = Boozer_I_j
            Boozer_G[js_b] = Boozer_G_j

            # Evaluate on this surface
            (
                bmnc_b_s,
                bmns_b_s,
                rmnc_b_s,
                rmns_b_s,
                zmnc_b_s,
                zmns_b_s,
                numnc_b_s,
                numns_b_s,
                gmnc_b_s,
                gmns_b_s,
            ) = jit_surface_solve(
                rmnc_j,
                rmns_j,
                zmnc_j,
                zmns_j,
                lmnc_j,
                lmns_j,
                bmnc_j,
                bmns_j,
                bsubumnc_j,
                bsubumns_j,
                bsubvmnc_j,
                bsubvmns_j,
                float(self.iota[js]),
                Boozer_I_j,
                Boozer_G_j,
            )

            # Store results into output arrays
            bmnc_b_out = bmnc_b_out.at[:, js_b].set(bmnc_b_s)
            rmnc_b_out = rmnc_b_out.at[:, js_b].set(rmnc_b_s)
            zmns_b_out = zmns_b_out.at[:, js_b].set(zmns_b_s)
            numns_b_out = numns_b_out.at[:, js_b].set(numns_b_s)
            gmnc_b_out = gmnc_b_out.at[:, js_b].set(gmnc_b_s)
            if self.asym:
                bmns_b_out = bmns_b_out.at[:, js_b].set(bmns_b_s)
                rmns_b_out = rmns_b_out.at[:, js_b].set(rmns_b_s)
                zmnc_b_out = zmnc_b_out.at[:, js_b].set(zmnc_b_s)
                numnc_b_out = numnc_b_out.at[:, js_b].set(numnc_b_s)
                gmns_b_out = gmns_b_out.at[:, js_b].set(gmns_b_s)

        # Save output arrays back to Python attributes as numpy arrays
        self.bmnc_b = _np.asarray(bmnc_b_out)
        self.gmnc_b = _np.asarray(gmnc_b_out)
        self.rmnc_b = _np.asarray(rmnc_b_out)
        self.zmns_b = _np.asarray(zmns_b_out)
        self.numns_b = _np.asarray(numns_b_out)
        self.Boozer_I = Boozer_I
        self.Boozer_G = Boozer_G
        if self.asym:
            self.bmns_b = _np.asarray(bmns_b_out)
            self.rmns_b = _np.asarray(rmns_b_out)
            self.zmnc_b = _np.asarray(zmnc_b_out)
            self.numnc_b = _np.asarray(numnc_b_out)
            self.gmns_b = _np.asarray(gmns_b_out)
        else:
            self.bmns_b = None
            self.rmns_b = None
            self.zmnc_b = None
            self.numnc_b = None
            self.gmns_b = None

        # Record the number of output surfaces and modes for external code
        self.ns_b = ns_b
        self.mnboz = mnboz

        # Save surfaces used for later reporting
        self.s_b = _np.asarray(self.s_in[self.compute_surfs])

        # Report if requested
        if self.verbose > 0:
            print(f"Completed Boozer transform on {ns_b} surfaces.")

    # ------------------------------------------------------------------
    def write_boozmn(self, filename: str) -> None:
        """Write the computed Boozer Fourier spectra to a ``boozmn`` NetCDF file.

        The ``boozmn`` format is used by the original ``booz_xform``
        package to store Boozer coordinates and related quantities.  This
        method reproduces the essential parts of the format.  See the
        booz_xform documentation for details.

        Parameters
        ----------
        filename : str
            Path of the output NetCDF file.
        """
        # Validate that run() has been called
        if self.bmnc_b is None:
            raise RuntimeError("run() must be called before write_boozmn()")
        # This method first attempts to use netCDF4; if unavailable
        # it falls back to SciPy's netcdf_file for NetCDF3 output.
        ns_in_plus_1 = self.ns_in + 1
        mnboz = self.mnboz
        ns_b = self.ns_b
        # Construct jlist (1-based indexing) for compute_surfs
        jlist = _np.array([idx + 2 for idx in self.compute_surfs], dtype='i4')
        # Prepare radial profiles with zero prepended
        iota_b = _np.zeros(ns_in_plus_1)
        buco_b = _np.zeros(ns_in_plus_1)
        bvco_b = _np.zeros(ns_in_plus_1)
        iota_b[1:] = _np.asarray(self.iota)
        buco_b[1:] = _np.asarray(self.Boozer_I_all)
        bvco_b[1:] = _np.asarray(self.Boozer_G_all)
        # Additional profiles
        def make_profile(arr):
            prof = _np.zeros(ns_in_plus_1)
            if arr is not None:
                prof[1:] = _np.asarray(arr)
            return prof
        profiles = {
            'phip_b': make_profile(self.phip),
            'chi_b': make_profile(self.chi),
            'pres_b': make_profile(self.pres),
            'phi_b': make_profile(self.phi),
        }
        # Spectral arrays need to be transposed to shape (pack_rad, mn_modes)
        bmnc_b = _np.asarray(self.bmnc_b).T
        rmnc_b = _np.asarray(self.rmnc_b).T
        zmns_b = _np.asarray(self.zmns_b).T
        pmns_b = -_np.asarray(self.numns_b).T
        gmn_b = _np.asarray(self.gmnc_b).T
        if self.asym:
            bmns_b = _np.asarray(self.bmns_b).T
            rmns_b = _np.asarray(self.rmns_b).T
            zmnc_b = _np.asarray(self.zmnc_b).T
            pmnc_b = -_np.asarray(self.numnc_b).T
            gmns_b = _np.asarray(self.gmns_b).T
        # Attempt to use netCDF4 for writing
        try:
            import netCDF4 as nc  # type: ignore
            ds = nc.Dataset(filename, 'w')
            # Define dimensions
            ds.createDimension('radius', ns_in_plus_1)
            ds.createDimension('mn_mode', mnboz)
            ds.createDimension('mn_modes', mnboz)
            ds.createDimension('comput_surfs', ns_b)
            ds.createDimension('pack_rad', ns_b)
            # Version and flags
            vvar = ds.createVariable('version', str)
            vvar[...] = _np.array(['JAX booz_xform'], dtype='object')
            asym_var = ds.createVariable('lasym__logical__', 'i4')
            asym_var[...] = 1 if self.asym else 0
            # Scalars
            def put_scalar(name, value):
                var = ds.createVariable(name, 'f8' if isinstance(value, float) else 'i4')
                var.assignValue(value)
            put_scalar('ns_b', int(self.ns_in + 1))
            put_scalar('nfp_b', int(self.nfp))
            put_scalar('mboz_b', int(self.mboz))
            put_scalar('nboz_b', int(self.nboz))
            put_scalar('mnboz_b', int(self.mnboz))
            put_scalar('aspect_b', float(self.aspect))
            # 1D arrays
            ds.createVariable('jlist', 'i4', ('comput_surfs',))[:] = jlist
            ds.createVariable('ixm_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xm_b, dtype='i4')
            ds.createVariable('ixn_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xn_b, dtype='i4')
            ds.createVariable('iota_b', 'f8', ('radius',))[:] = iota_b
            ds.createVariable('buco_b', 'f8', ('radius',))[:] = buco_b
            ds.createVariable('bvco_b', 'f8', ('radius',))[:] = bvco_b
            for name, data in profiles.items():
                ds.createVariable(name, 'f8', ('radius',))[:] = data
            # 2D arrays with dims (pack_rad, mn_modes)
            dims = ('pack_rad', 'mn_mode')
            ds.createVariable('bmnc_b', 'f8', dims)[:, :] = bmnc_b
            ds.createVariable('rmnc_b', 'f8', dims)[:, :] = rmnc_b
            ds.createVariable('zmns_b', 'f8', dims)[:, :] = zmns_b
            ds.createVariable('pmns_b', 'f8', dims)[:, :] = pmns_b
            ds.createVariable('gmn_b', 'f8', dims)[:, :] = gmn_b
            if self.asym:
                ds.createVariable('bmns_b', 'f8', dims)[:, :] = bmns_b
                ds.createVariable('rmns_b', 'f8', dims)[:, :] = rmns_b
                ds.createVariable('zmnc_b', 'f8', dims)[:, :] = zmnc_b
                ds.createVariable('pmnc_b', 'f8', dims)[:, :] = pmnc_b
                ds.createVariable('gmns_b', 'f8', dims)[:, :] = gmns_b
            ds.close()
            return
        except Exception:
            pass
        # Fallback: use SciPy to write a NetCDF3 file
        from scipy.io import netcdf_file  # type: ignore
        ds = netcdf_file(filename, 'w')
        # Define dimensions
        ds.createDimension('radius', ns_in_plus_1)
        ds.createDimension('mn_mode', mnboz)
        ds.createDimension('mn_modes', mnboz)
        ds.createDimension('comput_surfs', ns_b)
        ds.createDimension('pack_rad', ns_b)
        # Scalars
        version_var = ds.createVariable('version', 'c', ('version_length',)) if False else None
        # SciPy netcdf_file does not support string variables easily; store version as a char array
        # We'll write the version as characters in a dimension we create here.
        verstr = b'JAX booz_xform'
        ds.createDimension('version_length', len(verstr))
        version_var = ds.createVariable('version', 'c', ('version_length',))
        version_var[:] = list(verstr)
        lasym_var = ds.createVariable('lasym__logical__', 'i', ())
        lasym_var.assignValue(1 if self.asym else 0)
        # Scalar ints/floats
        def put_scalar_scipy(name, value):
            var = ds.createVariable(name, 'f8' if isinstance(value, float) else 'i', ())
            var.assignValue(value)
        put_scalar_scipy('ns_b', int(self.ns_in + 1))
        put_scalar_scipy('nfp_b', int(self.nfp))
        put_scalar_scipy('mboz_b', int(self.mboz))
        put_scalar_scipy('nboz_b', int(self.nboz))
        put_scalar_scipy('mnboz_b', int(self.mnboz))
        put_scalar_scipy('aspect_b', float(self.aspect))
        # 1D arrays
        jvar = ds.createVariable('jlist', 'i', ('comput_surfs',))
        jvar[:] = jlist
        xvar = ds.createVariable('ixm_b', 'i', ('mn_modes',))
        xvar[:] = _np.asarray(self.xm_b, dtype='i')
        nvar = ds.createVariable('ixn_b', 'i', ('mn_modes',))
        nvar[:] = _np.asarray(self.xn_b, dtype='i')
        # Radial profiles
        for name, data in [('iota_b', iota_b), ('buco_b', buco_b), ('bvco_b', bvco_b)]:
            var = ds.createVariable(name, 'f8', ('radius',))
            var[:] = data
        for name, data in profiles.items():
            var = ds.createVariable(name, 'f8', ('radius',))
            var[:] = data
        # 2D arrays dims (pack_rad, mn_mode)
        dims = ('pack_rad', 'mn_mode')
        for name, data in [
            ('bmnc_b', bmnc_b),
            ('rmnc_b', rmnc_b),
            ('zmns_b', zmns_b),
            ('pmns_b', pmns_b),
            ('gmn_b', gmn_b),
        ]:
            var = ds.createVariable(name, 'f8', dims)
            var[:, :] = data
        if self.asym:
            for name, data in [
                ('bmns_b', bmns_b),
                ('rmns_b', rmns_b),
                ('zmnc_b', zmnc_b),
                ('pmnc_b', pmnc_b),
                ('gmns_b', gmns_b),
            ]:
                var = ds.createVariable(name, 'f8', dims)
                var[:, :] = data
        ds.close()
        return

    # ------------------------------------------------------------------
    def read_boozmn(self, filename: str) -> None:
        """Read Boozer Fourier data from a ``boozmn`` NetCDF file.

        This method populates the attributes of this instance from a file
        produced by :meth:`write_boozmn` or the original ``booz_xform``
        program.  Existing data will be overwritten.

        Parameters
        ----------
        filename : str
            Path to a ``boozmn`` NetCDF file.
        """
        try:
            import netCDF4 as nc
        except ImportError as e:
            raise ImportError(
                "The netCDF4 package is required to read boozmn files. "
                "Install it via 'pip install netCDF4'"
            ) from e

        with nc.Dataset(filename, 'r') as ds:
            # Determine symmetry
            self.asym = bool(ds.variables['lasym__logical__'][...].item())
            # Dimensions
            ns_in_plus_1 = ds.dimensions['radius'].size
            self.ns_in = ns_in_plus_1 - 1
            self.ns_b = ds.dimensions['pack_rad'].size
            self.mnboz = ds.dimensions['mn_mode'].size
            # Scalars
            self.nfp = int(ds.variables['nfp_b'][...].item())
            self.mboz = int(ds.variables['mboz_b'][...].item())
            self.nboz = int(ds.variables['nboz_b'][...].item())
            # Read indices
            self.compute_surfs = [int(j - 2) for j in ds.variables['jlist'][:]]
            self.xm_b = _np.asarray(ds.variables['ixm_b'][:], dtype=int)
            self.xn_b = _np.asarray(ds.variables['ixn_b'][:], dtype=int)
            # Radial profiles
            iota_b = _np.asarray(ds.variables['iota_b'][:])
            self.iota = jnp.asarray(iota_b[1:], dtype=jnp.float64)
            buco_b = _np.asarray(ds.variables['buco_b'][:])
            bvco_b = _np.asarray(ds.variables['bvco_b'][:])
            self.Boozer_I_all = buco_b[1:]
            self.Boozer_G_all = bvco_b[1:]
            # Optional profiles
            for name, attr in [('phip_b','phip'), ('chi_b','chi'), ('pres_b','pres'), ('phi_b','phi')]:
                if name in ds.variables:
                    arr = _np.asarray(ds.variables[name][:])
                    setattr(self, attr, jnp.asarray(arr[1:], dtype=jnp.float64))
            # Spectra
            # Spectra are stored in the boozmn file with the radial index
            # (pack_rad dimension) as the first dimension and the mode index
            # (mn_mode dimension) as the second.  Internally we store
            # arrays as (mnboz, ns_b), so transpose here.
            self.bmnc_b = _np.asarray(ds.variables['bmnc_b'][:, :]).T
            self.rmnc_b = _np.asarray(ds.variables['rmnc_b'][:, :]).T
            self.zmns_b = _np.asarray(ds.variables['zmns_b'][:, :]).T
            self.numns_b = -_np.asarray(ds.variables['pmns_b'][:, :]).T
            self.gmnc_b = _np.asarray(ds.variables['gmn_b'][:, :]).T
            if self.asym:
                self.bmns_b = _np.asarray(ds.variables['bmns_b'][:, :]).T
                self.rmns_b = _np.asarray(ds.variables['rmns_b'][:, :]).T
                self.zmnc_b = _np.asarray(ds.variables['zmnc_b'][:, :]).T
                self.numnc_b = -_np.asarray(ds.variables['pmnc_b'][:, :]).T
                self.gmns_b = _np.asarray(ds.variables['gmns_b'][:, :]).T
            else:
                self.bmns_b = None
                self.rmns_b = None
                self.zmnc_b = None
                self.numnc_b = None
                self.gmns_b = None
            # Derive Boozer I and G for computed surfaces
            self.Boozer_I = _np.asarray(self.Boozer_I_all[self.compute_surfs])
            self.Boozer_G = _np.asarray(self.Boozer_G_all[self.compute_surfs])
            # s_b: if s_in exists from prior init, update; else create uniform grid
            if self.s_in is None:
                # s_in was not provided earlier; create uniform grid
                full_grid = jnp.linspace(0.0, 1.0, ns_in_plus_1, dtype=jnp.float64)
                self.s_in = full_grid[1:]
            # Surfaces used
            self.s_b = _np.asarray(self.s_in[self.compute_surfs])
