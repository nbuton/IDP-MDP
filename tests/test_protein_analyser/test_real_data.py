import pytest
import numpy as np
from idpmdp.protein_analyzer import ProteinAnalyzer
from pathlib import Path


# This fixture initializes the class for use in all test functions
@pytest.fixture
def analyzer():
    pdb_path = Path("data/IDRome/IDRome_v4/Q5/T7/B8/541_1292/top_AA.pdb")
    xtc_path = Path("data/IDRome/IDRome_v4/Q5/T7/B8/541_1292/traj_AA.xtc")
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


def test_gyration(analyzer):
    results = analyzer.compute_gyration_tensor_properties()
    n_frames = len(analyzer.u.trajectory)

    for key, value in results.items():
        # 1. Check consistency and type
        assert isinstance(value, np.ndarray), f"{key} is not a numpy array"
        assert (
            len(value) == n_frames
        ), f"Number of {key} ({len(value)}) != frames ({n_frames})"

        # 2. Key-specific physical checks
        if "relative" in key or "/" in key:
            # Ratios and Relative shapes (asphericity, anisotropy, etc.)
            # These should be positive and typically between 0 and 1
            assert np.all(value >= 0), f"{key} has negative values"
            assert np.all(value <= 1.0001), f"{key} exceeds 1.0"

        elif "eigenvalues" in key or "radius" in key:
            # Eigenvalues and Rg must be strictly positive
            assert np.all(value > 0), f"{key} must be strictly positive"

        elif key == "prolateness":
            # Prolateness can be negative (oblate) or positive (prolate)
            # We just check that it is within a reasonable range (usually -2 to 2)
            assert np.all(value >= -2.0) and np.all(
                value <= 2.0
            ), f"Prolateness {value} out of bounds"

    # 3. Hierarchy check (l1 <= l2 <= l3)
    # This ensures your eigenvalue sorting logic is correct
    l1 = results["gyration_eigenvalues_l1"]
    l2 = results["gyration_eigenvalues_l2"]
    l3 = results["gyration_eigenvalues_l3"]
    assert np.all(l1 <= l2), "Eigenvalue l1 > l2"
    assert np.all(l2 <= l3), "Eigenvalue l2 > l3"


def test_secondary_structure_propensities(analyzer):
    """Check if secondary structure propensities are computed correctly."""
    propensities = analyzer.compute_secondary_structure_propensities()

    assert isinstance(propensities, dict)

    all_2d_structure = ["H", "G", "I", "E", "B", "T", "S", "C"]
    num_residues = len(analyzer.residues)
    for sec_structure in all_2d_structure:
        assert len(propensities["ss_propensity_" + sec_structure]) == num_residues
        # Check that propensities are between 0 and 1
        assert np.all(
            (np.array(propensities["ss_propensity_" + sec_structure]) >= 0)
            & (np.array(propensities["ss_propensity_" + sec_structure]) <= 1)
        )


def test_scaling_exponent(analyzer):
    nu = analyzer.compute_scaling_exponent()
    assert 0 < nu < 1.0  # Physically realistic range for polymers


def test_secondary_structure(analyzer):
    ss = analyzer.compute_secondary_structure_propensities()

    # Ensure we have data to check
    assert len(ss) > 0, "Propensity dictionary is empty"

    # 1. Convert all dictionary values into a single 2D NumPy array
    # Shape will be (number_of_categories, protein_size)
    propensities_matrix = np.stack(list(ss.values()))

    # 2. Sum along axis 0 (summing categories for each residue)
    # This results in a 1D array of length 'protein_size'
    total_propensities = np.sum(propensities_matrix, axis=0)

    # 3. Verify every residue sums to 1.0
    # Using np.isclose handles floating point math errors (e.g., 0.99999999)
    assert np.all(
        np.isclose(total_propensities, 1.0)
    ), f"Sum of propensities is not 1.0. Totals found: {total_propensities}"

    # 4. Verify length matches protein size
    for key in ss:
        assert len(ss[key]) == analyzer.protein_size, f"Length mismatch in {key}"


def test_dihedrals_and_entropy(analyzer):
    data = analyzer.compute_dihedral_distribution()

    for angle in ["phi", "psi"]:
        assert angle in data
        values = data[angle]

        # 1. Check shape and type
        assert isinstance(values, np.ndarray)
        assert values.ndim == 2

        # 2. Physical Boundary Check (Ignoring NaNs)
        # np.nanmin/max ignore NaNs and return the min/max of the actual numbers
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        assert vmin >= -180.01, f"{angle} has values below -180: {vmin}"
        assert vmax <= 180.01, f"{angle} has values above 180: {vmax}"

        # 3. Verify NaNs exist where expected (Optional but Recommended)
        # For a single chain: phi[0] and psi[-1] are usually NaN
        if angle == "phi":
            assert np.isnan(values[:, 0]).all(), "First residue phi should be NaN"
        if angle == "psi":
            assert np.isnan(values[:, -1]).all(), "Last residue psi should be NaN"

    # 4. Entropy Check
    if "configurational_entropy" in data:
        s = data["configurational_entropy"]
        # If entropy is calculated per-residue, it might also have NaNs
        # Use np.nanmean or check scalar
        assert np.nanmin(s) >= 0, "Configurational entropy cannot be negative"


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


def test_sasa_dictionary_output(analyzer):
    stride = 10
    # The method now returns a dictionary of arrays
    sasa_results = analyzer.compute_residue_sasa(n_sphere_points=60, stride=stride)

    # 1. Define expected keys
    expected_keys = ["sasa_abs_mean", "sasa_abs_std", "sasa_rel_mean", "sasa_rel_std"]

    for key in expected_keys:
        assert key in sasa_results, f"Missing {key} in SASA output"
        values = sasa_results[key]

        # 2. Verify it's a numpy array of the correct size
        assert isinstance(values, np.ndarray), f"{key} should be a numpy array"
        assert (
            len(values) == analyzer.protein_size
        ), f"Length of {key} ({len(values)}) does not match protein size ({analyzer.protein_size})"

        # 3. Check for non-negative values
        assert np.all(values >= 0), f"Negative values found in {key}"

    # 4. Specific check for Relative SASA (0.0 to 1.0 range)
    # Relative SASA is the absolute area divided by the max possible area for that residue type
    rel_mean = sasa_results["sasa_rel_mean"]

    # Use 1.15 to account for terminal residues and
    # differences in probe-surface algorithms
    max_allowed_rsa = 1.15

    assert np.all(rel_mean <= max_allowed_rsa), (
        f"Relative SASA exceeds physical limit of {max_allowed_rsa}. "
        f"Max found: {np.max(rel_mean)} at index {np.argmax(rel_mean)}"
    )

    # 5. Check fluctuations (Standard Deviation)
    # If the protein is moving, the std should generally be > 0
    assert np.any(sasa_results["sasa_abs_std"] >= 0)
