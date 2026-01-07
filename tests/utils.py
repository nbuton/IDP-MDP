import MDAnalysis as mda
import os
from pathlib import Path


def write_dummy_files(u, tmp_path):
    """
    Writes a MDAnalysis Universe to both PDB and XTC formats.

    Args:
        u (mda.Universe): The universe object to save.

    Returns:
        tuple: (pdb_path, xtc_path)
    """

    pdb_path = os.path.join(tmp_path, "dummy.pdb")
    xtc_path = os.path.join(tmp_path, "dummy.xtc")

    # Write PDB (Standard text format)
    with mda.Writer(pdb_path) as W:
        W.write(u.atoms)

    # Write XTC (Binary trajectory format - requires n_atoms)
    with mda.Writer(xtc_path, n_atoms=u.atoms.n_atoms) as W:
        W.write(u.atoms)

    return Path(pdb_path), Path(xtc_path)
