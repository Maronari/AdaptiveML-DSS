from __future__ import annotations

import collections
import collections.abc

import numpy as np


def patch_numpy_for_lightautoml() -> None:
    """Restore numpy APIs expected by LightAutoML on NumPy 2.x."""
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "PINF"):
        np.PINF = np.inf
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf
    if hasattr(np, "find_common_type"):
        return

    def _find_common_type(array_types, scalar_types):
        if not array_types and not scalar_types:
            return np.dtype("float64")
        if scalar_types:
            scalar_dtypes = [np.asarray(value).dtype for value in scalar_types]
            return np.result_type(*array_types, *scalar_dtypes)
        return np.result_type(*array_types)

    np.find_common_type = _find_common_type


def patch_collections_for_experta() -> None:
    """Restore collections aliases expected by Experta on modern Python."""
    for name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
