"""Plotting utilities for the JAX version of the Boozer transform.

This module provides convenience functions for visualising the output of
the :class:`booz_xform_jax.booz_xform.BoozXform` class.  The API is
modelled on the plotting functions in the original ``booz_xform``
package.  All functions accept either a :class:`BoozXform` instance
or the path to a ``boozmn`` file.  Matplotlib is used for creating
figures.  Plotly support is not included in this version.

The following functions are provided:

``surfplot``
    Show the magnetic field strength |B| on a flux surface versus the
    Boozer poloidal and toroidal angles.

``symplot``
    Plot the radial variation of all Fourier modes of |B| grouped by
    symmetry type.

``modeplot``
    Plot the radial variation of the largest Fourier modes of |B|.

``wireplot``
    Make a 3D figure showing curves of constant Boozer angles on a flux
    surface.

These functions are optional; if matplotlib is not available they will
raise an exception at runtime.
"""

from __future__ import annotations

import numpy as np
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError as e:
    plt = None  # type: ignore

# Import the BoozXform class from the core module.  This import is
# deferred to support the modular structure of the package.
from .core import BoozXform


def handle_b_input(b: str | BoozXform) -> BoozXform:
    """Convert input to a :class:`BoozXform` instance.

    Parameters
    ----------
    b : str or BoozXform
        If a string, it is interpreted as a path to a ``boozmn`` file
        which will be read into a new :class:`BoozXform` instance.  If
        an instance of :class:`BoozXform` is provided it is returned
        unchanged.

    Returns
    -------
    BoozXform
        The corresponding BoozXform instance.
    """
    if isinstance(b, str):
        bx = BoozXform()
        bx.read_boozmn(b)
        return bx
    elif isinstance(b, BoozXform):
        return b
    else:
        raise ValueError("Argument must be a path to a boozmn file or a BoozXform instance")


def surfplot(
    b: str | BoozXform,
    js: int = 0,
    fill: bool = True,
    ntheta: int = 50,
    nphi: int = 90,
    ncontours: int = 25,
    **kwargs,
) -> None:
    """Plot |B| on a flux surface versus the Boozer angles.

    This function generates a 2D contour plot of the magnetic field
    strength |B| as a function of the Boozer toroidal angle ϕ and
    poloidal angle θ on the specified surface.

    Parameters
    ----------
    b : str or BoozXform
        The BoozXform instance to plot, or a filename of a ``boozmn`` file.
    js : int, optional
        Index among the output surfaces to plot (default is 0).
    fill : bool, optional
        Whether to fill contours (default True).  If False, line
        contours are drawn instead.
    ntheta : int, optional
        Number of grid points in the poloidal angle.
    nphi : int, optional
        Number of grid points in the toroidal angle.
    ncontours : int, optional
        Number of contour levels.
    **kwargs
        Additional keyword arguments passed to matplotlib's contour
        functions.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    b = handle_b_input(b)
    # Create meshgrid of Boozer angles
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d = np.linspace(0, 2 * np.pi / b.nfp, nphi)
    phi, theta = np.meshgrid(phi1d, theta1d)
    # Evaluate |B|
    modB = np.zeros_like(phi)
    for jmn in range(len(b.xm_b)):
        m = b.xm_b[jmn]
        n = b.xn_b[jmn]
        angle = m * theta - n * phi
        modB += b.bmnc_b[jmn, js] * np.cos(angle)
        if b.asym and b.bmns_b is not None:
            modB += b.bmns_b[jmn, js] * np.sin(angle)
    # Plot
    if fill:
        cs = plt.contourf(phi, theta, modB, ncontours, **kwargs)
    else:
        cs = plt.contour(phi, theta, modB, ncontours, **kwargs)
    plt.colorbar(cs)
    plt.xlabel(r'Boozer toroidal angle $\varphi$')
    plt.ylabel(r'Boozer poloidal angle $\theta$')
    plt.title(f'|B| on surface s={float(b.s_b[js]):.4f}')


def symplot(
    b: str | BoozXform,
    max_m: int = 20,
    max_n: int = 20,
    ymin: float | None = None,
    sqrts: bool = False,
    log: bool = True,
    B0: bool = True,
    helical_detail: bool = False,
    legend_args: dict | None = None,
    **kwargs,
) -> None:
    """Plot radial variation of all Fourier modes of |B| grouped by symmetry.

    The modes are coloured according to whether m=0, n=0, or both.  Modes
    with |m| and |n| exceeding the specified maxima are omitted.

    Parameters
    ----------
    b : str or BoozXform
        The BoozXform instance to plot, or a filename of a ``boozmn`` file.
    max_m : int
        Maximum poloidal mode number to include.
    max_n : int
        Maximum toroidal mode number (divided by nfp) to include.
    ymin : float, optional
        Lower limit for the y axis in logarithmic plots.  If not
        specified, a value is chosen automatically.
    sqrts : bool
        Whether to plot sqrt(s) on the x axis instead of s.
    log : bool
        Use a logarithmic y axis.
    B0 : bool
        Include the (m,n)=(0,0) mode in the figure.
    helical_detail : bool
        Whether to show details of helical modes separately.
    legend_args : dict, optional
        Additional arguments to pass to ``plt.legend``.
    **kwargs
        Additional keyword arguments passed to ``plt.plot``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    b = handle_b_input(b)
    legend_args = legend_args or {}
    if ymin is None:
        ymin = np.max(b.bmnc_b) * 1e-4
    mnmax = len(b.xm_b)
    rad = np.sqrt(b.s_b) if sqrts else b.s_b
    def my_abs(x):
        return np.abs(x) if log else x
    # Colours
    background_color = 'b'
    QA_color = [0, 0.7, 0]
    mirror_color = [0.7, 0.5, 0]
    helical_color = [1, 0, 1]
    helical_plus_color = 'gray'
    helical_minus_color = 'c'
    # Plot one example of each group for legend
    # (m,n)=(0,0)
    for imode in range(mnmax):
        if b.xm_b[imode] == 0 and b.xn_b[imode] == 0:
            if B0:
                plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=background_color,
                         label='m=0, n=0', **kwargs)
            break
    # (m≠0,n=0)
    for imode in range(mnmax):
        if b.xm_b[imode] != 0 and b.xn_b[imode] == 0:
            plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=QA_color,
                     label='m≠0, n=0', **kwargs)
            break
    # (m=0,n≠0)
    for imode in range(mnmax):
        if b.xm_b[imode] == 0 and b.xn_b[imode] != 0:
            plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=mirror_color,
                     label='m=0, n≠0', **kwargs)
            break
    # Helical modes for legend
    if helical_detail:
        for imode in range(mnmax):
            if b.xm_b[imode] != 0 and b.xn_b[imode] == b.xm_b[imode] * b.nfp:
                plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=helical_plus_color,
                         label='n=nfp*m', **kwargs)
                break
        for imode in range(mnmax):
            if b.xm_b[imode] != 0 and b.xn_b[imode] == -b.xm_b[imode] * b.nfp:
                plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=helical_minus_color,
                         label='n=-nfp*m', **kwargs)
                break
        for imode in range(mnmax):
            if b.xm_b[imode] != 0 and b.xn_b[imode] != 0 \
               and b.xn_b[imode] != b.xm_b[imode] * b.nfp \
               and b.xn_b[imode] != -b.xm_b[imode] * b.nfp:
                plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=helical_color,
                         label='other helical', **kwargs)
                break
    else:
        for imode in range(mnmax):
            if b.xm_b[imode] != 0 and b.xn_b[imode] != 0 and \
               b.xn_b[imode] != b.xm_b[imode] * b.nfp and b.xn_b[imode] != -b.xm_b[imode] * b.nfp:
                plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=helical_color,
                         label='helical', **kwargs)
                break
    plt.legend(**legend_args)
    # Now plot all modes
    for imode in range(mnmax):
        m = b.xm_b[imode]
        n = b.xn_b[imode] // b.nfp if b.nfp != 0 else 0
        if abs(m) > max_m or abs(n) > max_n:
            continue
        if not B0 and m == 0 and n == 0:
            continue
        # Determine colour
        if n == 0:
            if m == 0:
                mycolor = background_color
            else:
                mycolor = QA_color
        else:
            if m == 0:
                mycolor = mirror_color
            else:
                if helical_detail:
                    if n == m:
                        mycolor = helical_plus_color
                    elif n == -m:
                        mycolor = helical_minus_color
                    else:
                        mycolor = helical_color
                else:
                    mycolor = helical_color
        plt.plot(rad, my_abs(b.bmnc_b[imode, :]), color=mycolor, **kwargs)
    # Axes labels
    if sqrts:
        plt.xlabel('$r/a$ = sqrt(normalized toroidal flux)')
    else:
        plt.xlabel('$s$ = normalized toroidal flux')
    plt.title('Fourier harmonics of |B| in Boozer coordinates')
    plt.xlim([0, 1])
    if log:
        plt.yscale('log')
        plt.gca().set_ylim(bottom=ymin)


def modeplot(
    b: str | BoozXform,
    nmodes: int = 10,
    ymin: float | None = None,
    sqrts: bool = False,
    log: bool = True,
    B0: bool = True,
    legend_args: dict | None = None,
    **kwargs,
) -> None:
    """Plot the radial variation of the largest Fourier modes of |B|.

    Modes are sorted by amplitude at the outermost computed surface and
    the largest ``nmodes`` are displayed.

    Parameters
    ----------
    b : str or BoozXform
        The BoozXform instance or path to a boozmn file.
    nmodes : int, optional
        Number of modes to display.
    ymin : float, optional
        Lower limit for the y axis in logarithmic plots.  If not
        specified, a value is chosen automatically.
    sqrts : bool
        Whether to plot sqrt(s) on the x axis instead of s.
    log : bool
        Use a logarithmic y axis.
    B0 : bool
        Whether to include the (m,n)=(0,0) mode.
    legend_args : dict, optional
        Additional arguments passed to ``plt.legend``.
    **kwargs
        Additional keyword arguments passed to ``plt.plot``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    b = handle_b_input(b)
    legend_args = legend_args or {}
    if ymin is None:
        ymin = np.max(b.bmnc_b) * 1e-4
    # Remove (m,n)=(0,0) if requested
    assert b.xm_b[0] == 0 and b.xn_b[0] == 0
    if B0:
        data = b.bmnc_b
        xm = b.xm_b
        xn = b.xn_b
    else:
        data = b.bmnc_b[1:, :]
        xm = b.xm_b[1:]
        xn = b.xn_b[1:]
    # Sort modes by amplitude at outermost radius
    sorted_indices = np.argsort(-np.abs(data[:, -1]))
    indices = sorted_indices[:nmodes]
    rad = np.sqrt(b.s_b) if sqrts else b.s_b
    def my_abs(x):
        return np.abs(x) if log else x
    if not log:
        plt.plot([0,1],[0,0], ':k')
    for index in indices:
        plt.plot(rad, my_abs(data[index, :]),
                 label=f'm={xm[index]}, n={xn[index]}', **kwargs)
    plt.legend(**legend_args)
    if sqrts:
        plt.xlabel('$r/a$ = sqrt(normalized toroidal flux)')
    else:
        plt.xlabel('$s$ = normalized toroidal flux')
    plt.title('Largest Fourier harmonics of |B| in Boozer coordinates')
    plt.xlim([0,1])
    if log:
        plt.yscale('log')
        plt.gca().set_ylim(bottom=ymin)


def wireplot(
    b: str | BoozXform,
    js: int | None = None,
    ntheta: int = 30,
    nphi: int = 80,
    refine: int = 3,
    surf: bool = True,
    orig: bool = True,
) -> None:
    """Make a 3D wireframe of curves of constant Boozer angles.

    This function recreates the 3D wireframe plot from the original
    ``booz_xform`` code.  Curves of constant Boozer poloidal angle θ
    and constant toroidal angle ϕ are drawn on a specified flux
    surface.  Optionally, the original VMEC coordinate curves are
    overlayed.  Requires ``matplotlib`` with 3D support.

    Parameters
    ----------
    b : str or BoozXform
        The BoozXform instance or path to a boozmn file.
    js : int, optional
        The index among the output surfaces to plot.  If None, the
        outermost surface is selected.
    ntheta : int, optional
        Number of curves in the poloidal angle.
    nphi : int, optional
        Number of curves in the toroidal angle.
    refine : int, optional
        Number of interpolation points between contours.  A higher
        value yields smoother curves.
    surf : bool, optional
        Whether to display a semi‑transparent surface representing
        the flux surface.
    orig : bool, optional
        Whether to overlay curves of constant VMEC angles.
    """
    if plt is None:
        raise RuntimeError("matplotlib with 3D support is required for wireplot")
    b = handle_b_input(b)
    # Choose surface
    if js is None:
        js = b.ns_b - 1
    # Real-space grid for angles
    ntheta0 = ntheta * refine + 1
    nphi0 = nphi * refine + 1
    theta1d = np.linspace(0, 2 * np.pi, ntheta0)
    phi1d = np.linspace(0, 2 * np.pi, nphi0)
    varphi, theta = np.meshgrid(phi1d, theta1d)
    # Initialise arrays
    R = np.zeros_like(theta)
    Z = np.zeros_like(theta)
    dR_dtheta = np.zeros_like(theta)
    dZ_dtheta = np.zeros_like(theta)
    nu = np.zeros_like(theta)
    # Accumulate Fourier sums
    for jmn in range(b.mnboz):
        m = b.xm_b[jmn]
        n = b.xn_b[jmn]
        angle = m * theta - n * varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        R += b.rmnc_b[jmn, js] * cosangle
        Z += b.zmns_b[jmn, js] * sinangle
        dR_dtheta += -m * b.rmnc_b[jmn, js] * sinangle
        dZ_dtheta += m * b.zmns_b[jmn, js] * cosangle
        nu += b.numns_b[jmn, js] * sinangle
        if b.asym and b.bmns_b is not None:
            R += b.rmns_b[jmn, js] * sinangle
            Z += b.zmnc_b[jmn, js] * cosangle
            nu += b.numnc_b[jmn, js] * cosangle
    # Convert Boozer toroidal angle to VMEC toroidal angle
    phi_vmec = varphi - nu
    # Compute cylindrical coordinates to Cartesian for plotting
    x = R * np.cos(phi_vmec)
    y = R * np.sin(phi_vmec)
    z = Z
    # Prepare figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Draw surface
    if surf:
        ax.plot_surface(x, y, z, alpha=0.3, linewidth=0, antialiased=False)
    # Draw Boozer coordinate curves
    # Poloidal curves: constant varphi index
    phi_indices = np.linspace(0, nphi0 - 1, nphi, dtype=int)
    for idx in phi_indices:
        ax.plot(x[:, idx], y[:, idx], z[:, idx], color='k')
    # Toroidal curves: constant theta index
    theta_indices = np.linspace(0, ntheta0 - 1, ntheta, dtype=int)
    for idx in theta_indices:
        ax.plot(x[idx, :], y[idx, :], z[idx, :], color='k')
    # Optionally overlay original VMEC coordinate curves
    if orig:
        # Original angles are simply theta and varphi; convert to Cartesian
        x0 = R * np.cos(varphi)
        y0 = R * np.sin(varphi)
        # Poloidal curves
        for idx in phi_indices:
            ax.plot(x0[:, idx], y0[:, idx], z[:, idx], color='r', linestyle='--')
        # Toroidal curves
        for idx in theta_indices:
            ax.plot(x0[idx, :], y0[idx, :], z[idx, :], color='r', linestyle='--')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Boozer coordinate wireframe on surface')
    # Return the figure so that callers can further customise or close it
    return fig