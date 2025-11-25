"""Compatibility wrapper for the BoozXform class.

This module exists to preserve the import path
``booz_xform_jax.booz_xform`` used by earlier versions of the
package.  It re‑exports the :class:`~booz_xform_jax.core.BoozXform`
class so that existing code continues to function.  New development
should import :class:`BoozXform` from :mod:`booz_xform_jax.core` or
directly from the top‑level :mod:`booz_xform_jax` package.
"""

from .core import BoozXform

__all__ = ['BoozXform']