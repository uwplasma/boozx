#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_li383_resolution_scan.py
================================

Resolution scan in (mboz, nboz) for the LI383 configuration.

This script:
  * reads a VMEC wout file,
  * runs the Boozer transform several times with increasing (mboz, nboz),
  * plots the |B| Fourier spectrum using symplot at each resolution.

The goal is to provide intuition for:
  * how many Boozer modes are needed for convergence,
  * how the various symmetry groups (m=0, n=0, helical, etc.)
    are represented in the Boozer spectrum.

Usage
-----

From the repository top-level:

    python -m examples.example_li383_resolution_scan \
        --mboz-list 8 12 16 \
        --nboz-list 8 12 16

If you run without options you get a small default scan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from booz_xform_jax.core import Booz_xform
from booz_xform_jax import plots


def find_default_wout() -> Path:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    return repo_root / "tests" / "test_files" / "wout_li383_1.4m.nc"


def parse_int_list(values: Sequence[str]) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boozer resolution scan in (mboz, nboz) for LI383."
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=find_default_wout(),
        help="Path to VMEC wout file (default: tests/test_files/wout_li383_1.4m.nc)",
    )
    parser.add_argument(
        "--mboz-list",
        nargs="+",
        default=["8", "12"],
        help="List of mboz values to scan (default: 8 12).",
    )
    parser.add_argument(
        "--nboz-list",
        nargs="+",
        default=["8", "12"],
        help="List of nboz values to scan (default: 8 12).",
    )
    parser.add_argument(
        "--s",
        type=float,
        default=0.5,
        help="Normalized toroidal flux (0..1) for surface to register (default: 0.5).",
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

    mboz_list = parse_int_list(args.mboz_list)
    nboz_list = parse_int_list(args.nboz_list)
    if len(mboz_list) != len(nboz_list):
        raise ValueError("mboz-list and nboz-list must have the same length.")

    print(f"[example_li383_resolution_scan] Using wout file: {wout_path}")
    print(f"[example_li383_resolution_scan] mboz_list = {mboz_list}")
    print(f"[example_li383_resolution_scan] nboz_list = {nboz_list}")

    # We will reuse the *same* VMEC data for each resolution, so we read
    # the wout file just once and reconfigure (mboz, nboz) in-place.
    bx = Booz_xform()
    bx.read_wout(str(wout_path), flux=False)
    bx.register_surfaces(args.s)

    fig, axes = plt.subplots(
        1, len(mboz_list), figsize=(5 * len(mboz_list), 4), squeeze=False
    )
    axes = axes[0]

    for i, (mboz, nboz) in enumerate(zip(mboz_list, nboz_list, strict=False)):
        print(f"[example_li383_resolution_scan] Running with mboz={mboz}, nboz={nboz}")
        bx.mboz = mboz
        bx.nboz = nboz
        bx.mnboz = None  # force recomputation of mode lists/grids
        bx.xm_b = None
        bx.xn_b = None
        bx._prepared = False

        bx.verbose = 0
        bx.run()

        ax = axes[i]
        plots.symplot(
            bx,
            max_m=mboz,
            max_n=nboz,
            sqrts=True,
            log=True,
            B0=True,
            helical_detail=False,
            legend_args={"fontsize": "x-small"},
            ax=ax,
            show=False,
        )
        ax.set_title(f"mboz={mboz}, nboz={nboz}")

    fig.tight_layout()
    fig.suptitle("|B| Fourier spectrum vs resolution", y=1.02)
    fig.tight_layout()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
