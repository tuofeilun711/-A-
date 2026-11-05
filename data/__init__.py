"""Data package: adds the bundled .pylibs directory (h5py) to sys.path.

Keeping the bootstrap here means every entry point that imports data.* works
even when h5py is not installed system-wide.
"""
import os as _os
import sys as _sys

_PYLIB = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '.pylibs')
if _os.path.isdir(_PYLIB) and _PYLIB not in _sys.path:
    _sys.path.insert(0, _PYLIB)
