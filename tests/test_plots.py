#!/usr/bin/env python3

"""Smoke tests for the plotting functions.

This test mirrors ``tests/test_plots.py`` from the original project.  It
exercises the four plotting routines (``surfplot``, ``symplot``,
``modeplot`` and ``wireplot``) in a non‑interactive manner.  The goal is
to ensure that these functions execute without raising exceptions when
supplied either a filename or a pre‑computed ``BoozXform`` instance.
Because ``plt.show()`` is not called, no figures are displayed on
screen.
"""

import os
import unittest
import matplotlib.pyplot as plt
import booz_xform_jax as bx

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


class PlotTest(unittest.TestCase):
    def test_calling_plots(self) -> None:
        """Call each plotting function with a variety of inputs."""
        configurations = [
            'circular_tokamak',
            'up_down_asymmetric_tokamak',
            'li383_1.4m',
            'LandremanSenguptaPlunk_section5p3',
        ]
        for config in configurations:
            for which_input in [0, 1]:
                if which_input == 0:
                    # Supply a filename string
                    b = os.path.join(TEST_DIR, f'boozmn_{config}.nc')
                else:
                    # Supply a BoozXform instance used to drive the transformation
                    wout_filename = os.path.join(TEST_DIR, f'wout_{config}.nc')
                    b = bx.BoozXform()
                    b.read_wout(wout_filename)
                    b.compute_surfs = [2, 15]
                    b.run()
                # Try different options for each plot type
                bx.surfplot(b)
                bx.surfplot(b, js=1, fill=False, cmap=plt.cm.jet)
                bx.symplot(b)
                bx.symplot(b, marker='.', sqrts=True, log=False)
                bx.modeplot(b)
                bx.modeplot(b, marker='.', sqrts=True, log=False,
                            legend_args={'fontsize': 6})
                # wireplot returns a figure; we call it to ensure no error
                fig = bx.wireplot(b)
                # It is safe to close the figure since we do not display it
                plt.close(fig)


if __name__ == '__main__':
    unittest.main()
