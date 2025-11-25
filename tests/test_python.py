#!/usr/bin/env python3

"""Basic API tests for the JAX implementation of Booz_xform.

This test file mirrors ``tests/test_python.py`` from the original
``booz_xform`` repository.  It verifies that simple attribute
assignments on the BoozXform object behave as expected.  The test
does not perform any numerical transformation.
"""

import os
import numpy as np
import unittest

from booz_xform_jax import BoozXform

# Alias for backwards compatibility with the original test code
Booz_xform = BoozXform

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


class MainTest(unittest.TestCase):
    def test_compute_surfs_edit(self) -> None:
        """Ensure that the compute_surfs property can be set and retrieved."""
        b = Booz_xform()
        b.read_wout(os.path.join(TEST_DIR, 'wout_li383_1.4m.nc'))
        # assign two surfaces and check they are stored unchanged
        b.compute_surfs = [10, 15]
        np.testing.assert_allclose(b.compute_surfs, [10, 15])


if __name__ == '__main__':
    unittest.main()
