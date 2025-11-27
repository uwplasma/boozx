#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_li383_wireframe.py
==========================

3D wireframe example for the LI383 configuration.

This script:
  * reads a VMEC wout file,
  * computes Boozer harmonics on a chosen surface,
  * uses wireplot to display curves of constant Boozer angles on that
    flux surface, optionally overlaying VMEC-angle curves.

The resulting figure is similar to the classic Boozer-coordinate
wireframe plots in the original booz_xform package.

Usage
-----

From the repository top-level:

    python -m examples.example_li383_wireframe

or

    python examples/example_li383_wireframe.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from booz_xform_jax.core import Booz_xform
from booz_xform_jax import plots


def find_default_wout() -> Path:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    return repo_root / "tests" / "test_files" / "wout_li383_1.4m.nc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D Boozer coordinate wireframe for LI383."
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=find_default_wout(),
        help="Path to VMEC wout file (default: tests/test_files/wout_li383_1.4m.nc)",
    )
    parser.add_argument(
        "--s",
        type=float,
        default=0.8,
        help="Normalized toroidal flux (0..1) for surface to plot (default: 0.8).",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=20,
        help="Number of Boozer poloidal curves (default: 20).",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=40,
        help="Number of Boozer toroidal curves (default: 40).",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=3,
        help="Interpolation factor between grid points (default: 3).",
    )
    parser.add_argument(
        "--no-orig",
        action="store_true",
        help="Do not overlay VMEC-angle curves.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure (useful in batch mode).",
    )
    args = parser.parse_args()

    wout_path = args.wout
    if not wout_path.is_file():
        raise FileNotFoundError(f"Could not find wout file: {wout_path}")

    print(f"[example_li383_wireframe] Using wout file: {wout_path}")

    # Compute Boozer harmonics on the requested surface.
    bx = Booz_xform()
    bx.read_wout(str(wout_path), flux=False)
    bx.register_surfaces(args.s)
    bx.verbose = 0
    bx.run()

    # The registered surface will be the last one in s_b / compute_surfs,
    # so js = ns_b - 1 selects it.
    js = bx.ns_b - 1
    print(
        f"[example_li383_wireframe] Plotting surface js={js} with s={float(bx.s_b[js]):.4f}"
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plots.wireplot(
        bx,
        js=js,
        ntheta=args.ntheta,
        nphi=args.nphi,
        refine=args.refine,
        surf=True,
        orig=not args.no_orig,
        ax=ax,
        show=False,   # we will handle plt.show() ourselves
    )

    fig.tight_layout()
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
