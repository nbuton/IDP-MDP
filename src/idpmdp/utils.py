import numpy as np
import diptest


def test_multimodality(data, alpha=0.05):
    """
    Tests for multimodality using Hartigan's Dip Test.
    Compatible with the current PyPI 'diptest' package.
    """
    # Ensure data is a 1D numpy array of floats
    data = np.sort(np.array(data).astype(float))
    data = data.reshape(-1)  # Ensure it's a 2D array for diptest

    # The function is named diptest.diptest()
    # It returns (dip_statistic, p_value)
    dip_stat, p_val = diptest.diptest(data)

    is_multimodal = p_val < alpha

    return dip_stat, p_val, is_multimodal
