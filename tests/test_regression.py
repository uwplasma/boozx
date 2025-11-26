#!/usr/bin/env python3

"""Regression tests comparing to reference boozmn files.

This test replicates ``tests/test_regression.py`` from the original
project.  It loads a precomputed boozmn file, then runs the
transformation on the corresponding VMEC ``wout`` file using the
parameters stored in the boozmn file.  The resulting spectra and
profiles are compared against the reference to ensure that the JAX
implementation reproduces the original results to high precision.

The reference boozmn files are stored in ``tests/test_files`` and
correspond to several different configurations (tokamak and
stellarator).  Both stellarator symmetric and asymmetric cases are
tested.  The optional flux arrays are exercised by toggling the
``flux`` argument in :meth:`read_wout`.
"""

import os
import unittest
import numpy as np
from scipy.io import netcdf_file

from booz_xform_jax import BoozXform

# Provide backwards compatibility alias
Booz_xform = BoozXform

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


class RegressionTest(unittest.TestCase):
    def test_regression(self) -> None:
        configurations = [
            'circular_tokamak',
            'up_down_asymmetric_tokamak',
            'li383_1.4m',
            'LandremanSenguptaPlunk_section5p3',
        ]
        for config in configurations:
            wout_filename = os.path.join(TEST_DIR, f'wout_{config}.nc')
            boozmn_filename = os.path.join(TEST_DIR, f'boozmn_{config}.nc')
            f = netcdf_file(boozmn_filename, 'r', mmap=False)
            for flux in [True, False]:
                b = Booz_xform()
                b.read_wout(wout_filename, flux=flux)
                # Transfer parameters from the reference file to the JAX calculation
                b.mboz = int(f.variables['mboz_b'][()])
                b.nboz = int(f.variables['nboz_b'][()])
                # jlist in boozmn files uses 1‑based indexing offset by +1 for the axis
                jlist = f.variables['jlist'][()]
                b.compute_surfs = [int(j) - 2 for j in jlist]
                # Run the transform
                b.run()
                # Determine which arrays to compare
                vars_to_compare = ['bmnc_b', 'rmnc_b', 'zmns_b', 'numns_b', 'gmnc_b']
                asym = bool(f.variables['lasym__logical__'][()])
                if asym:
                    vars_to_compare += ['bmns_b', 'rmns_b', 'zmnc_b', 'numnc_b', 'gmns_b']
                rtol = 1e-12
                atol = 1e-12
                for var in vars_to_compare:
                    # gmnc_b is misspelled as gmn_b in the reference files
                    var_ref_name = 'gmn_b' if var == 'gmnc_b' else var
                    # For nu arrays the boozmn format stores p = -nu
                    sign = -1 if var.startswith('numn') else 1
                    arr_ref = f.variables[var_ref_name][:]
                    arr_new = getattr(b, var).T  # b stores (mnboz, ns_b)
                    # Check maximum difference for debugging
                    diff = np.max(np.abs(arr_ref - sign * arr_new))
                    print(f'max absolute diff in {var}:', diff)
                    np.testing.assert_allclose(sign * arr_new, arr_ref, rtol=rtol, atol=atol)
                # Compare select variables written to boozmn files
                boozmn_new_filename = os.path.join(TEST_DIR, f'boozmn_new_{config}.nc')
                b.write_boozmn(boozmn_new_filename)
                f2 = netcdf_file(boozmn_new_filename)
                vars_in_file = f.variables.keys()
                # Variables excluded because they depend on optional flux profiles
                if flux:
                    exclude = ['rmax_b', 'rmin_b', 'betaxis_b', 'version', 'beta_b']
                else:
                    exclude = ['rmax_b', 'rmin_b', 'betaxis_b', 'version', 'pres_b', 'beta_b', 'phip_b', 'phi_b']
                for var in vars_in_file:
                    if var in exclude:
                        continue
                    arr_ref = f.variables[var][:]
                    arr_new = f2.variables[var][:]
                    diff = np.max(np.abs(arr_ref - arr_new))
                    print(f'max absolute diff in {var}:', diff)
                    np.testing.assert_allclose(arr_new, arr_ref, rtol=rtol, atol=atol)
                # Clean up temporary file
                f2.close()
                os.remove(boozmn_new_filename)
            f.close()


if __name__ == '__main__':
    unittest.main()
