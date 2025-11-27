"""Compatibility wrapper for the Booz_xform class.

This module exists to preserve the import path
``booz_xform_jax.booz_xform`` used by earlier versions of the
package.  It re‑exports the :class:`~booz_xform_jax.core.Booz_xform`
class so that existing code continues to function.  New development
should import :class:`Booz_xform` from :mod:`booz_xform_jax.core` or
directly from the top‑level :mod:`booz_xform_jax` package.
"""

from .core import Booz_xform

__all__ = ['Booz_xform']