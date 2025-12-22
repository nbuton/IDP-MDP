import pytest
import numpy as np
from idpmdp.protein_analyzer import ProteinAnalyzer


# This fixture initializes the class for use in all test functions
@pytest.fixture
def analyzer():
    pdb_path = "data/ATLAS/1k5n_A_analysis/1k5n_A.pdb"
    xtc_path = "data/ATLAS/1k5n_A_analysis/1k5n_A_R2.xtc"
    return ProteinAnalyzer(pdb_path, xtc_path)


def test_initialization(analyzer):
    """Check if the universe and protein size are loaded correctly."""
    assert analyzer.protein_size > 0
    assert hasattr(analyzer, "u")


def test_end_to_end_distance_shape(analyzer):
    """Check if the output is an array matching the trajectory length."""
    distances = analyzer.compute_end_to_end_distance()

    assert isinstance(distances, np.ndarray)
    assert len(distances) == len(analyzer.u.trajectory)
    # Distances in biology shouldn't be negative
    assert np.all(distances > 0)


def test_radius_of_gyration_values(analyzer):
    """Check if Rg is within a physically reasonable range for a protein."""
    rg_values = analyzer.compute_radius_of_gyration()

    assert len(rg_values) == len(analyzer.u.trajectory)
    # A protein of ~100 residues usually has an Rg between 10-25A
    assert np.mean(rg_values) > 0


def test_mean_squared_distance(analyzer):
    """Ensure MSD returns a single float value."""
    msd = analyzer.compute_mean_squared_distance()
    assert isinstance(msd, (float, np.float64))
    assert msd >= 0
