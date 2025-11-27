#!/usr/bin/env python3

"""Test round‑trip writing and reading of boozmn files.

This test mirrors ``tests/test_write_read.py`` from the original
repository.  It writes a boozmn file using the JAX implementation,
then reads that file back into a new object and compares all
quantities to ensure they match the original.
"""

import os
import numpy as np
import unittest

from booz_xform_jax import Booz_xform

# Alias for compatibility with the original test names
Booz_xform = Booz_xform

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


class WriteReadTest(unittest.TestCase):
    def test_write_read(self) -> None:
        """Write a boozmn file and read it back into a new object."""
        configurations = [
            'circular_tokamak',
            'up_down_asymmetric_tokamak',
            'li383_1.4m',
            'LandremanSenguptaPlunk_section5p3',
        ]
        for configuration in configurations:
            for compute_surfs in [[0], [0, 5], [5, 10, 15]]:
                wout_filename = 'wout_' + configuration + '.nc'
                boozmn_filename = os.path.join(TEST_DIR, f'boozmn_new_{configuration}.nc')
                b1 = Booz_xform()
                # Test both flux and non‑flux reading
                for flux in [True, False]:
                    b1.read_wout(os.path.join(TEST_DIR, wout_filename), flux)
                    b1.compute_surfs = compute_surfs
                    b1.run()
                    b1.write_boozmn(boozmn_filename)
                    # Read back into a new object
                    b2 = Booz_xform()
                    b2.read_boozmn(boozmn_filename)
                    # Simple scalar properties
                    self.assertEqual(b1.asym, b2.asym)
                    self.assertEqual(b1.nfp, b2.nfp)
                    self.assertEqual(b1.mboz, b2.mboz)
                    self.assertEqual(b1.nboz, b2.nboz)
                    self.assertEqual(b1.mnboz, b2.mnboz)
                    self.assertEqual(b1.ns_b, b2.ns_b)
                    np.testing.assert_equal(b1.xm_b, b2.xm_b)
                    np.testing.assert_equal(b1.xn_b, b2.xn_b)
                    np.testing.assert_equal(b1.compute_surfs, b2.compute_surfs)
                    # Tolerances for numerical comparisons
                    rtol = 1e-12
                    atol = 1e-12
                    # Spectral data
                    for name in [
                        'bmnc_b',
                        'rmnc_b',
                        'zmns_b',
                        'numns_b',
                        'gmnc_b',
                    ]:
                        arr1 = getattr(b1, name)
                        arr2 = getattr(b2, name)
                        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
                    if b1.asym:
                        for name in [
                            'bmns_b',
                            'rmns_b',
                            'zmnc_b',
                            'numnc_b',
                            'gmns_b',
                        ]:
                            arr1 = getattr(b1, name)
                            arr2 = getattr(b2, name)
                            np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
                    # Radial profiles
                    np.testing.assert_allclose(b1.s_b, b2.s_b, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(b1.iota, b2.iota, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(b1.Boozer_G_all, b2.Boozer_G_all, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(b1.Boozer_I_all, b2.Boozer_I_all, rtol=rtol, atol=atol)
                    if flux:
                        np.testing.assert_allclose(b1.phip, b2.phip, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(b1.chi, b2.chi, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(b1.pres, b2.pres, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(b1.phi, b2.phi, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(b1.toroidal_flux, b2.toroidal_flux, rtol=rtol, atol=atol)
                    else:
                        # When flux data are not read, these arrays are empty or zero
                        self.assertTrue(b1.phip is None or len(b1.phip) == 0)
                        self.assertTrue(b2.phip is None or b2.phip == 0)
                        self.assertTrue(b1.chi is None or len(b1.chi) == 0)
                        self.assertTrue(b2.chi is None or b2.chi == 0)
                        self.assertTrue(b1.pres is None or len(b1.pres) == 0)
                        self.assertTrue(b2.pres is None or b2.pres == 0)
                        self.assertTrue(b1.phi is None or len(b1.phi) == 0)
                        self.assertTrue(b2.phi is None or b2.phi == 0)
                        self.assertEqual(b1.toroidal_flux, 0)
                        self.assertEqual(b2.toroidal_flux, 0)
                    # Cleanup
                    os.remove(boozmn_filename)


if __name__ == '__main__':
    unittest.main()
