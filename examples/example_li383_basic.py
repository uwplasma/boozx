#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_li383_basic.py
======================

Basic end-to-end example using booz_xform_jax:

  * read a VMEC wout file for the LI383 configuration,
  * compute Boozer harmonics on a single flux surface,
  * generate:
       - a |B|(theta_B, phi_B) surfplot,
       - a modeplot of the largest Boozer Fourier modes.

This script is intentionally pedagogical and small. It is a good place
to start if you are new to the JAX version of booz_xform.

Usage
-----

From the repository top-level:

    python -m examples.example_li383_basic

or

    python examples/example_li383_basic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from booz_xform_jax.core import Booz_xform
from booz_xform_jax import plots


def find_default_wout() -> Path:
    """
    Return the default wout path used in this example.

    We assume the standard repository layout:

        repo_root/
          tests/
            test_files/
              wout_li383_1.4m.nc

    and build the path relative to this script.
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    return repo_root / "tests" / "test_files" / "wout_li383_1.4m.nc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basic booz_xform_jax example on LI383."
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
        default=0.5,
        help="Normalized toroidal flux value (0..1) of the surface to transform "
             "(default: 0.5). The closest VMEC half-grid surface will be used.",
    )
    parser.add_argument(
        "--mboz",
        type=int,
        default=None,
        help="Maximum Boozer poloidal mode index m. "
             "If not given, defaults to mpol from the VMEC file.",
    )
    parser.add_argument(
        "--nboz",
        type=int,
        default=None,
        help="Maximum Boozer toroidal period index n. "
             "If not given, defaults to ntor from the VMEC file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures (useful in batch mode).",
    )
    args = parser.parse_args()

    wout_path = args.wout
    if not wout_path.is_file():
        raise FileNotFoundError(f"Could not find wout file: {wout_path}")

    print(f"[example_li383_basic] Using wout file: {wout_path}")

    # ------------------------------------------------------------------
    # 1) Read VMEC data and initialise the Booz_xform instance
    # ------------------------------------------------------------------
    bx = Booz_xform()
    # The 'flux=True' flag also reads the flux profiles (phi, phip, etc.)
    bx.read_wout(str(wout_path), flux=True)

    # Optionally override Boozer resolution:
    if args.mboz is not None:
        bx.mboz = args.mboz
    if args.nboz is not None:
        bx.nboz = args.nboz

    # ------------------------------------------------------------------
    # 2) Register a single surface in normalized toroidal flux s
    # ------------------------------------------------------------------
    # register_surfaces accepts either indices or s values in [0,1].
    bx.register_surfaces(args.s)

    # ------------------------------------------------------------------
    # 3) Run the Boozer transform
    # ------------------------------------------------------------------
    bx.verbose = 1
    bx.run()

    print("[example_li383_basic] Transform complete.")
    print(f"  nfp     = {bx.nfp}")
    print(f"  ns_in   = {bx.ns_in}")
    print(f"  ns_b    = {bx.ns_b}")
    print(f"  mnboz   = {bx.mnboz}")
    print(f"  s_b[0]  = {float(bx.s_b[0]):.4f}")
    print(f"  Boozer I= {float(bx.Boozer_I[0]):.6e}")
    print(f"  Boozer G= {float(bx.Boozer_G[0]):.6e}")

    # ------------------------------------------------------------------
    # 4) Make a |B|(theta_B, phi_B) surfplot for this surface
    # ------------------------------------------------------------------
    print("[example_li383_basic] Generating surfplot...")
    plots.surfplot(
        bx,
        js=0,
        ntheta=80,
        nphi=160,
        ncontours=40,
        cmap="viridis",
        show=not args.no_show,
        savefig=None,  # change to "li383_surfplot.png" to save
    )

    # ------------------------------------------------------------------
    # 5) Make a modeplot of the largest harmonics at all transformed radii
    # ------------------------------------------------------------------
    print("[example_li383_basic] Generating modeplot...")
    plots.modeplot(
        bx,
        nmodes=10,
        sqrts=True,
        log=True,
        B0=True,
        legend_args={"loc": "best", "fontsize": "small"},
        show=not args.no_show,
        savefig=None,  # change to "li383_modeplot.png" to save
    )

    print("[example_li383_basic] Done.")


if __name__ == "__main__":
    main()
