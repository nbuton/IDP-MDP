import pytest
import numpy as np
from idpmdp.analysis.orchestrator import ProteinAnalyzer
import MDAnalysis as mda
from tests.utils import write_dummy_files
from idpmdp.analysis.global_metrics import (
    compute_end_to_end_distance,
)


def create_dummy_protein(ca1_coords, ca2_coords):
    """
    Creates a 3-residue Glycine chain.
    Hardcodes the first and last C-alpha, randomizes everything else.
    """
    n_residues = 3
    n_atoms = n_residues * 7

    # 1. Define the names
    atomnames = ["N", "CA", "C", "O", "H", "HA1", "HA2"] * n_residues

    # 2. Map names to elements (N->N, CA->C, C->C, O->O, H/HA->H)
    # We strip the numbers from the names to get the element
    elements = []
    for name in atomnames:
        if name.startswith("CA"):
            elements.append("C")
        elif name.startswith("C"):
            elements.append("C")
        elif name.startswith("N"):
            elements.append("N")
        elif name.startswith("O"):
            elements.append("O")
        elif name.startswith("H"):
            elements.append("H")
        else:
            elements.append("X")  # Fallback

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=np.repeat(range(3), 7),
        trajectory=True,
    )

    # 3. Add the missing attributes
    u.add_TopologyAttr("name", atomnames)
    u.add_TopologyAttr("element", elements)  # This fixes your error
    u.add_TopologyAttr("resname", ["GLY"] * n_residues)
    u.add_TopologyAttr("resid", [1, 2, 3])
    u.add_TopologyAttr("chainID", ["A"] * n_atoms)
    u.add_TopologyAttr("segid", ["SYSTEM"])
    u.add_TopologyAttr("occupancies", [1.0] * n_atoms)
    u.add_TopologyAttr("tempfactors", [0.0] * n_atoms)
    u.add_TopologyAttr("record_types", ["ATOM"] * n_atoms)
    u.add_TopologyAttr("altLocs", [" "] * n_atoms)  # Single space string
    u.add_TopologyAttr("icodes", [" "] * n_residues)  # Single space string
    u.add_TopologyAttr("formalcharges", [0] * n_atoms)  # Integers

    # Initialize random coordinates
    coords = np.random.random((n_atoms, 3)).astype(np.float32) * 10

    # Hardcode the first CA (index 1) and last CA (index 15)
    # Indices in a 7-atom GLY: 0:N, 1:CA, 2:C... 7:N, 8:CA... 14:N, 15:CA
    u.dimensions = [100, 100, 100, 90, 90, 90]
    coords = np.random.random((n_atoms, 3)).astype(np.float32) * 10
    u.atoms.positions = coords
    u.atoms[1].position = ca1_coords
    u.atoms[15].position = ca2_coords

    return u


@pytest.mark.parametrize(
    "ca1, ca2, expected_dist",
    [
        ([0, 0, 0], [1, 1, 1], np.sqrt(3)),
        (
            [2.3, 5.6, -0.12],
            [-3.4, -1, 7.3],
            np.linalg.norm(np.array([2.3, 5.6, -0.12]) - np.array([-3.4, -1, 7.3])),
        ),
    ],
)
def test_end_to_end_distance(ca1, ca2, expected_dist, tmp_path):
    # Repeat the randomness 5 times to ensure inner atom positions don't affect CA-CA distance
    for i in range(5):
        u = create_dummy_protein(ca1, ca2)

        iteration_dir = tmp_path / f"iteration_{i}"
        iteration_dir.mkdir()  # This creates the folder on disk
        pdb_path, xtc_path = write_dummy_files(u, iteration_dir)

        analyzer = ProteinAnalyzer(pdb_path, xtc_path)
        dist = compute_end_to_end_distance(analyzer.md_analysis_u, analyzer.residues)

        # Assert with floating point tolerance
        assert dist == pytest.approx(expected_dist, rel=1e-5)
