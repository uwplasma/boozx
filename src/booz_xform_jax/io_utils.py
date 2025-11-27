"""Input/output utilities for the JAX implementation of ``booz_xform``.

This module provides functions to read from and write to the NetCDF
formats used by the original ``booz_xform`` code.  In particular it
handles the ``boozmn`` file format, which stores Boozer‐coordinate
Fourier spectra and associated radial profiles.  These functions
operate on :class:`~booz_xform_jax.core.Booz_xform` instances and are
used by the methods :meth:`booz_xform_jax.core.Booz_xform.write_boozmn`
and :meth:`booz_xform_jax.core.Booz_xform.read_boozmn`.
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

def write_boozmn(self, filename: str) -> None:
    """Write the computed Boozer Fourier spectra to a ``boozmn`` NetCDF file.

    The ``boozmn`` format is used by the original ``booz_xform`` package to
    store Boozer coordinates and related quantities.  This function
    writes the essential information required to reconstruct the Boozer
    harmonics: the mode definitions, radial profiles, and spectral
    coefficients.  If the instance was initialised from a VMEC
    equilibrium and :meth:`booz_xform_jax.core.Booz_xform.run` has been
    called, this routine will create a NetCDF file that can be read by
    the original ``booz_xform`` or by :func:`read_boozmn` below.

    Parameters
    ----------
    self : Booz_xform
        The instance containing the results of the Boozer transform.
    filename : str
        Path of the output NetCDF file.
    """
    # Ensure run() has been called
    if self.bmnc_b is None:
        raise RuntimeError("run() must be called before write_boozmn()")
    if self.verbose > 0:
        print(f"[booz_xform_jax] Writing boozmn file: {filename}")
    # Prepare sizes
    ns_in_plus_1 = int(self.ns_in) + 1
    mnboz = int(self.mnboz)
    ns_b = int(self.ns_b)
    # Construct jlist: convert compute_surfs to 1-based full-grid indices
    jlist = _np.array([idx + 2 for idx in self.compute_surfs], dtype='i4')
    # Prepare radial profiles with zero prepended
    iota_b = _np.zeros(ns_in_plus_1)
    buco_b = _np.zeros(ns_in_plus_1)
    bvco_b = _np.zeros(ns_in_plus_1)
    iota_b[1:] = _np.asarray(self.iota)
    buco_b[1:] = _np.asarray(self.Boozer_I_all)
    bvco_b[1:] = _np.asarray(self.Boozer_G_all)
    # Helper to build profiles only if present
    profiles: dict[str, _np.ndarray] = {}
    def add_profile(name: str, arr: Optional[jnp.ndarray]) -> None:
        if arr is not None:
            prof = _np.zeros(ns_in_plus_1)
            prof[1:] = _np.asarray(arr)
            profiles[name] = prof
    add_profile('phip_b', self.phip)
    add_profile('chi_b',  self.chi)
    add_profile('pres_b', self.pres)
    add_profile('phi_b',  self.phi)
    # Spectral arrays need to be transposed to shape (pack_rad, mn_mode)
    bmnc_b = _np.asarray(self.bmnc_b).T
    rmnc_b = _np.asarray(self.rmnc_b).T
    zmns_b = _np.asarray(self.zmns_b).T

    # Parallel current harmonics: pmns_b is minus numns_b, as in the original code
    numns_b = _np.asarray(self.numns_b).T
    pmns_b = -numns_b

    gmn_b = _np.asarray(self.gmnc_b).T
    if self.asym:
        bmns_b = _np.asarray(self.bmns_b).T
        rmns_b = _np.asarray(self.rmns_b).T
        zmnc_b = _np.asarray(self.zmnc_b).T

        numnc_b = _np.asarray(self.numnc_b).T
        pmnc_b = -numnc_b

        gmns_b = _np.asarray(self.gmns_b).T
    # Attempt to use netCDF4 for writing; fall back to SciPy if necessary
    try:
        import netCDF4 as nc  # type: ignore
        ds = nc.Dataset(filename, 'w')
        using_netcdf4 = True
    except Exception:
        using_netcdf4 = False
    if not using_netcdf4:
        # SciPy netcdf_file writes NetCDF3 format
        if 'scipy.io' not in globals():
            from scipy.io import netcdf_file  # type: ignore
        ds = netcdf_file(filename, 'w')  # type: ignore
    # Define dimensions
    ds.createDimension('radius', ns_in_plus_1)
    ds.createDimension('mn_mode', mnboz)
    ds.createDimension('mn_modes', mnboz)
    ds.createDimension('comput_surfs', ns_b)
    ds.createDimension('pack_rad', ns_b)
    # Write version and symmetry flag
    if using_netcdf4:
        vvar = ds.createVariable('version', str)
        vvar[...] = _np.array(['JAX booz_xform'], dtype='object')
        asym_var = ds.createVariable('lasym__logical__', 'i4')
        asym_var[...] = 1 if self.asym else 0
    else:
        ds.version = 'JAX booz_xform'
        ds.lasym__logical__ = 1 if self.asym else 0
    # Helper to write scalars
    def put_scalar(name: str, value):
        if using_netcdf4:
            var = ds.createVariable(name, 'f8' if isinstance(value, float) else 'i4')
            var.assignValue(value)
        else:
            setattr(ds, name, value)
    # ns_b in the original boozmn files is the number of VMEC radial grid points
    # (the full "radius" grid, length ns_in + 1), *not* the number of packed
    # Boozer surfaces (pack_rad). Use ns_in_plus_1 here to match the C++ booz_xform.
    put_scalar('ns_b', int(ns_in_plus_1))
    put_scalar('nfp_b', int(self.nfp))
    put_scalar('mboz_b', int(self.mboz))
    put_scalar('nboz_b', int(self.nboz))
    put_scalar('mnboz_b', int(self.mnboz))
    put_scalar('aspect_b', float(self.aspect))
    # Store toroidal flux for round-trip consistency
    put_scalar('toroidal_flux_b', float(self.toroidal_flux))
    # Write 1D arrays
    if using_netcdf4:
        ds.createVariable('jlist', 'i4', ('comput_surfs',))[:] = jlist
        ds.createVariable('ixm_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xm_b, dtype='i4')
        ds.createVariable('ixn_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xn_b, dtype='i4')
        ds.createVariable('iota_b', 'f8', ('radius',))[:] = iota_b
        ds.createVariable('buco_b', 'f8', ('radius',))[:] = buco_b
        ds.createVariable('bvco_b', 'f8', ('radius',))[:] = bvco_b
        for name, data in profiles.items():
            ds.createVariable(name, 'f8', ('radius',))[:] = data
    else:
        ds.createVariable('jlist', 'i4', ('comput_surfs',))[:] = jlist
        ds.createVariable('ixm_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xm_b, dtype='i4')
        ds.createVariable('ixn_b', 'i4', ('mn_modes',))[:] = _np.asarray(self.xn_b, dtype='i4')
        ds.createVariable('iota_b', 'f8', ('radius',))[:] = iota_b
        ds.createVariable('buco_b', 'f8', ('radius',))[:] = buco_b
        ds.createVariable('bvco_b', 'f8', ('radius',))[:] = bvco_b
        for name, data in profiles.items():
            ds.createVariable(name, 'f8', ('radius',))[:] = data
    # Write 2D arrays: dims (pack_rad, mn_mode)
    dims = ('pack_rad', 'mn_mode')
    ds.createVariable('bmnc_b', 'f8', dims)[:, :] = bmnc_b
    ds.createVariable('rmnc_b', 'f8', dims)[:, :] = rmnc_b
    ds.createVariable('zmns_b', 'f8', dims)[:, :] = zmns_b

    # Parallel current and its minus sign variant (pmns_b = -numns_b)
    ds.createVariable('numns_b', 'f8', dims)[:, :] = numns_b
    ds.createVariable('pmns_b', 'f8', dims)[:, :] = pmns_b

    ds.createVariable('gmn_b', 'f8', dims)[:, :] = gmn_b

    if self.asym:
        ds.createVariable('bmns_b', 'f8', dims)[:, :] = bmns_b
        ds.createVariable('rmns_b', 'f8', dims)[:, :] = rmns_b
        ds.createVariable('zmnc_b', 'f8', dims)[:, :] = zmnc_b

        ds.createVariable('numnc_b', 'f8', dims)[:, :] = numnc_b
        ds.createVariable('pmnc_b', 'f8', dims)[:, :] = pmnc_b

        ds.createVariable('gmns_b', 'f8', dims)[:, :] = gmns_b
    # Close file
    ds.close()
    if self.verbose > 0:
        print(f"[booz_xform_jax] Finished writing {filename}")
    return None


def read_boozmn(self, filename: str) -> None:
    """Read Boozer Fourier data from a ``boozmn`` NetCDF file.

    This routine populates a :class:`~booz_xform_jax.core.Booz_xform`
    instance with data from a file produced by the original
    ``booz_xform`` program or by :func:`write_boozmn`.  It reads the
    mode definitions, radial profiles and spectral arrays, reorienting
    the latter into the internal ``(mnboz, ns_b)`` layout.  Existing
    data on the instance will be overwritten.

    Parameters
    ----------
    self : Booz_xform
        The instance to populate.
    filename : str
        Path to a ``boozmn`` NetCDF file.
    """
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The netCDF4 package is required to read boozmn files. "
            "Install it via 'pip install netCDF4'"
        ) from e
    if self.verbose > 0:
        print(f"[booz_xform_jax] Reading boozmn file: {filename}")
    with nc.Dataset(filename, 'r') as ds:
        # Symmetry flag
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
        # Toroidal flux (if present – older files may not have it)
        if 'toroidal_flux_b' in ds.variables:
            self.toroidal_flux = float(ds.variables['toroidal_flux_b'][...].item())
        else:
            self.toroidal_flux = 0.0
        # Indices of selected surfaces (convert from 1-based jlist)
        self.compute_surfs = [int(j) - 2 for j in ds.variables['jlist'][:]]
        # Mode lists
        self.xm_b = _np.asarray(ds.variables['ixm_b'][:], dtype=int)
        self.xn_b = _np.asarray(ds.variables['ixn_b'][:], dtype=int)
        # Radial profiles on full grid
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
        # Spectra: stored as (pack_rad, mn_mode)
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
        # Derive Boozer I and G on selected surfaces
        self.Boozer_I = _np.asarray(self.Boozer_I_all)[self.compute_surfs]
        self.Boozer_G = _np.asarray(self.Boozer_G_all)[self.compute_surfs]
        # If s_in is not already defined (i.e., we haven't run init_from_vmec),
        # reconstruct the same half-grid used in init_from_vmec:
        if self.s_in is None:
            full_grid = _np.linspace(0.0, 1.0, ns_in_plus_1)  # 0..1, length ns_in+1
            half_grid = 0.5 * (full_grid[:-1] + full_grid[1:])  # midpoints
            self.s_in = half_grid
        # Set s_b for selected surfaces
        self.s_b = _np.asarray(self.s_in)[self.compute_surfs]
    if self.verbose > 0:
        print("[booz_xform_jax] Finished reading boozmn file")
    return None