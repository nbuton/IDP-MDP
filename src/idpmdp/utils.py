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


def get_ensemble_summary(
    data_dict, include_min_max=False, include_histogram=False, bins=20
):
    """
    Summarizes time-series data into ensemble-wide descriptors.

    By default, this method computes the arithmetic mean and standard deviation
    for every property provided in the data_dict,

    Args:
        data_dict (dict): Dictionary of property arrays.
        include_min_max (bool): If True, returns min and max values.
        include_histogram (bool): If True, returns counts and bin edges.
        bins (int/str): Number of bins or method (e.g., 'auto') for np.histogram.
    """
    summary = {}

    for key, values in data_dict.items():
        data = np.array(values)

        # Core Stats
        summary[f"{key}_mean"] = np.mean(data)
        summary[f"{key}_std"] = np.std(data)

        # Optional: Range
        if include_min_max:
            summary[f"{key}_min"] = np.min(data)
            summary[f"{key}_max"] = np.max(data)

        # Optional: Distribution
        if include_histogram:
            # density=True gives the probability density instead of raw counts
            counts, bin_edges = np.histogram(data, bins=bins, density=True)
            summary[f"{key}_hist_counts"] = counts
            summary[f"{key}_hist_edges"] = bin_edges

    return summary
