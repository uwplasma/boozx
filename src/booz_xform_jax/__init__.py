"""Top level package for the JAX implementation of BOOZ_XFORM.

This package exposes a :class:`~booz_xform_jax.booz_xform.Booz_xform` class
that mirrors the API of the original C++ implementation but is written
entirely in Python and uses JAX for array operations, JIT compilation and
automatic differentiation.  See the documentation in
``booz_xform_jax.booz_xform`` for more details.
"""

# Re-export the Booz_xform class from the core module.  This alias
# maintains backwards compatibility with earlier versions that
# defined Booz_xform in the top-level ``booz_xform`` module.  Users
# should import Booz_xform from ``booz_xform_jax`` directly.
from .core import Booz_xform
from .plots import surfplot, symplot, modeplot, wireplot

# Version is defined here for convenience.  It must match the value in
# pyproject.toml for consistency.  Updating the version in one place
# should be mirrored in the other.
__version__ = "0.1.0"

__all__ = [
    "Booz_xform",
    "surfplot",
    "symplot",
    "modeplot",
    "wireplot",
    "__version__",
]