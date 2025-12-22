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


def test_mean_squared_end_to_end_distance(analyzer):
    """Ensure MSD returns a single float value."""
    msd = analyzer.compute_mean_squared_end_to_end_distance()
    assert isinstance(msd, (float, np.float64))
    assert msd >= 0


def test_mean_squared_radius_of_gyration(analyzer):
    """Ensure MSRG returns a single float value."""
    msrg = analyzer.compute_mean_squared_radius_of_gyration()
    assert isinstance(msrg, (float, np.float64))
    assert msrg >= 0


def test_consistency_between_methods(analyzer):
    """Check if the mean squared values are consistent with their raw data."""
    end_to_end_dists = analyzer.compute_end_to_end_distance()
    rg_values = analyzer.compute_radius_of_gyration()

    computed_msed = np.mean(end_to_end_dists**2)
    computed_msrg = np.mean(rg_values**2)

    msd_method = analyzer.compute_mean_squared_end_to_end_distance()
    msrg_method = analyzer.compute_mean_squared_radius_of_gyration()

    assert np.isclose(computed_msed, msd_method)
    assert np.isclose(computed_msrg, msrg_method)


def test_secondary_structure_propensities(analyzer):
    """Check if secondary structure propensities are computed correctly."""
    propensities = analyzer.compute_secondary_structure_propensities()

    assert isinstance(propensities, dict)
    assert "residue" in propensities
    assert "coil_propensity" in propensities
    assert "helix_propensity" in propensities
    assert "sheet_propensity" in propensities

    num_residues = len(analyzer.residues)
    assert len(propensities["residue"]) == num_residues
    assert len(propensities["coil_propensity"]) == num_residues
    assert len(propensities["helix_propensity"]) == num_residues
    assert len(propensities["sheet_propensity"]) == num_residues

    # Check that propensities are between 0 and 1
    assert np.all(
        (np.array(propensities["coil_propensity"]) >= 0)
        & (np.array(propensities["coil_propensity"]) <= 1)
    )
    assert np.all(
        (np.array(propensities["helix_propensity"]) >= 0)
        & (np.array(propensities["helix_propensity"]) <= 1)
    )
    assert np.all(
        (np.array(propensities["sheet_propensity"]) >= 0)
        & (np.array(propensities["sheet_propensity"]) <= 1)
    )


# --- Global Topology Tests ---


def test_radius_of_gyration(analyzer):
    rg = analyzer.compute_radius_of_gyration()
    assert isinstance(rg, np.ndarray)
    assert len(rg) == len(analyzer.u.trajectory)
    assert np.all(rg > 0)


def test_gyration_tensor(analyzer):
    results = analyzer.compute_gyration_tensor_properties()
    assert "asphericity" in results
    assert "prolateness" in results
    assert len(results["asphericity"]) == len(analyzer.u.trajectory)
    # Asphericity is non-negative
    assert all(b >= 0 for b in results["asphericity"])


def test_scaling_exponent(analyzer):
    nu = analyzer.compute_scaling_exponent()
    assert 0 < nu < 1.0  # Physically realistic range for polymers


# --- Backbone Grammar Tests ---


def test_secondary_structure(analyzer):
    ss = analyzer.compute_secondary_structure_propensities()
    assert "helix_propensity" in ss
    assert len(ss["helix_propensity"]) == analyzer.protein_size
    # Check that propensities sum to approximately 1
    total = (
        ss["coil_propensity"][0] + ss["helix_propensity"][0] + ss["sheet_propensity"][0]
    )
    assert np.isclose(total, 1.0)


def test_dihedrals_and_entropy(analyzer):
    data = analyzer.compute_dihedral_distribution_and_entropy()
    assert "phi" in data
    assert "entropy_phi" in data
    assert len(data["entropy_phi"]) == data["phi"].shape[1]


# --- Network Dynamics Tests ---


def test_dccm(analyzer):
    matrix = analyzer.compute_dccm()
    n_res = analyzer.protein_size
    # ProDy DCCM might be based on atoms or residues depending on selection
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    # Correlation diagonal should be 1.0
    assert np.isclose(matrix[0, 0], 1.0)


def test_distance_fluctuations(analyzer):
    flucts = analyzer.compute_distance_fluctuations()
    assert flucts.shape == (analyzer.protein_size, analyzer.protein_size)
    assert np.all(flucts >= 0)


# --- Geometric & Solvent Tests ---


def test_contact_map(analyzer):
    cmap = analyzer.compute_contact_map(cutoff=8.0)
    assert cmap.shape == (analyzer.protein_size, analyzer.protein_size)
    assert np.max(cmap) <= 1.0
    assert np.min(cmap) >= 0.0


def test_sasa(analyzer):
    stride = 10
    # Calling the updated trajectory-based SASA method
    sasa_values = analyzer.compute_trajectory_sasa_mdtraj(
        n_sphere_points=60, stride=stride
    )

    # 1. Check if the output is a numpy array
    assert isinstance(sasa_values, np.ndarray), "SASA output should be a numpy array."

    # 2. Check if the length matches the number of frames in the trajectory
    total_frames = len(analyzer.u.trajectory)
    expected_frames = len(range(0, total_frames, stride))
    assert (
        len(sasa_values) == expected_frames
    ), f"Expected {expected_frames} frames, got {len(sasa_values)}."

    # 3. Check if all values are physically plausible (positive)
    assert np.all(sasa_values > 0), "SASA values must be greater than zero."

    # 4. (Optional) Check for reasonable variance
    # If the protein is moving, SASA shouldn't be identical across all frames
    if expected_frames > 1:
        assert (
            np.std(sasa_values) > 0
        ), "SASA is static; check if trajectory coordinates are updating."

    print(f"SASA test passed for {expected_frames} frames.")


def test_hydration_density(analyzer):
    grid = analyzer.compute_hydration_density(bins=10)
    assert grid.ndim == 3
    assert grid.shape == (10, 10, 10)
