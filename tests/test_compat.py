import numpy as np

from backend.utils.compat import patch_numpy_for_lightautoml


def test_patch_numpy_for_lightautoml_restores_numpy2_aliases():
    patch_numpy_for_lightautoml()

    assert np.NaN is np.nan
    assert np.Inf == np.inf
    assert np.PINF == np.inf
    assert np.NINF == -np.inf
    assert callable(np.find_common_type)
