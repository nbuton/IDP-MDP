import MDAnalysis
import mdtraj as md
from idpmdp.utils import get_ensemble_summary, mean_std
from idpmdp.analysis.global_metrics import (
    compute_end_to_end_distance,
    compute_gyration_tensor_properties,
    compute_maximum_diameter,
    compute_scaling_exponent,
)
from idpmdp.analysis.residue_level_metrics import (
    compute_residue_sasa,
    compute_secondary_structure_propensities,
    pooled_dihedral_entropy,
    compute_local_chirality,
)
from idpmdp.analysis.matrix_metrics import (
    compute_contact_map,
    compute_dccm,
    compute_distance_fluctuations,
)
import tarfile
import tempfile
from pathlib import Path
import numpy as np
import logging


class ProteinAnalyzer:
    def __init__(self, pdb_path, xtc_path=None, from_PED=False):
        """Initializes the Universe and checks the system size."""
        assert pdb_path.exists(), f"Expected file {pdb_path} to exist, but it does not."

        if from_PED:
            self.pdb_path = self._prepare_pdb_path(pdb_path)
        else:
            self.pdb_path = pdb_path

        self.xtc_path = xtc_path

        topology = pdb_path
        trajectory = xtc_path  # This can be None, a string, or a list

        if trajectory is None:
            self.md_analysis_u = MDAnalysis.Universe(topology)
            self.md_traj = md.load(self.pdb_path)
            print(f"Loaded Universe with {len(self.md_analysis_u.trajectory)} frames.")
            assert (
                len(self.md_analysis_u.trajectory) > 1
            ), "Be carefull no trajectory is loaded. Delete this raise if you known what you do"

        else:
            assert (
                xtc_path.exists()
            ), f"Expected file {xtc_path} to exist, but it does not."
            self.md_analysis_u = MDAnalysis.Universe(topology, trajectory)
            self.md_traj = md.load(self.xtc_path, top=self.pdb_path)

        n_chains = len(set(self.md_analysis_u.atoms.chainIDs))
        print(f"Number of chains: {n_chains}")
        assert n_chains == 1, "There are multiple chains in this pdb"

        # Select only protein atoms
        protein_selection = self.md_analysis_u.select_atoms("protein")
        self.md_analysis_u.transfer_to_memory(atomgroup=protein_selection)
        print(f"Total frames captured: {self.md_analysis_u.trajectory.n_frames}")

        assert check_atom_consistency(self.md_analysis_u)
        assert hasattr(self.md_analysis_u.atoms, "elements")
        assert hasattr(self.md_analysis_u.atoms, "masses")
        # Verify that the universe contains only one segment
        assert (
            len(self.md_analysis_u.segments) == 1
        ), "The provided PDB file contains multiple segments."

        # Identify the protein and its size
        heavy_atoms = self.md_analysis_u.select_atoms("not element H")
        self.is_coarse_grained = len(heavy_atoms) < (
            len(self.md_analysis_u.residues) * 4
        )
        self.protein_atoms = self.md_analysis_u.select_atoms("protein")
        self.residues = self.protein_atoms.residues
        self.protein_size = len(self.residues)

        print(f"Loaded Universe with {len(self.md_analysis_u.trajectory)} frames.")
        print(f"Protein size: {self.protein_size} residues.")

    def _prepare_pdb_path(self, path):
        # Simple check for GZIP magic number 0x1f 0x8b
        with open(path, "rb") as f:
            header = f.read(2)

        if header == b"\x1f\x8b":
            print(f"Detected archive: {path.name}. Extracting...")
            self.temp_dir = tempfile.TemporaryDirectory()

            with tarfile.open(path, "r:gz") as tar:
                # Find the actual PDB file inside the tarball
                members = [m for m in tar.getmembers() if m.name.endswith(".pdb")]
                if not members:
                    raise FileNotFoundError("No .pdb file found inside the archive.")

                # Extract the first PDB found
                tar.extract(members[0], path=self.temp_dir.name)
                return Path(self.temp_dir.name) / members[0].name

        return path

    def compute_all(
        self,
        sasa_n_sphere=960,
        scaling_min_sep=5,
        contact_cutoff=8.0,
    ):
        """
        Executes all analysis methods and aggregates results into a single dictionary.

        Returns:
            dict: Comprehensive analysis results.
        """
        results = {}
        logging.debug("Start computing end to end distances")
        results["avg_squared_Ree"], results["std_squared_Ree"] = mean_std(
            compute_end_to_end_distance(self.md_analysis_u, self.residues), squared=True
        )
        logging.debug("Start computing maximum diameter")
        results["avg_maximum_diameter"], results["std_maximum_diameter"] = mean_std(
            compute_maximum_diameter(self.md_analysis_u), squared=False
        )
        logging.debug("Start computing gyration tensor")
        gyration_output = compute_gyration_tensor_properties(self.md_traj)
        results.update(
            get_ensemble_summary(
                gyration_output, include_min_max=False, include_histogram=False, bins=20
            )
        )

        logging.debug("Start computing scalling exponent")
        results["scaling_exponent"] = compute_scaling_exponent(
            self.md_traj, min_sep=scaling_min_sep
        )

        logging.debug("Start computing local chirality")
        # results["chimical_shift"] = compute_chemical_shift(self.md_traj) # Need the conda env of LEGOLAS to be use -> So computed in another script afterward
        results["local_chirality"] = compute_local_chirality(self.md_traj)
        results.update(compute_secondary_structure_propensities(self.md_traj))

        logging.debug("Start computing DCCM")
        results["dccm"] = compute_dccm(self.md_analysis_u, self.protein_atoms)
        results.update(pooled_dihedral_entropy(self.md_traj, bins=60))

        logging.debug("Start computing distance fluctuation")
        results["distance_fluctuations"] = compute_distance_fluctuations(
            self.md_analysis_u, self.protein_atoms
        )
        logging.debug("Start computing contact map frequency")
        results["contact_map"] = compute_contact_map(
            self.md_analysis_u, self.protein_atoms, cutoff=contact_cutoff
        )

        logging.debug("Start computing SASA")
        results.update(
            compute_residue_sasa(self.md_traj, n_sphere_points=sasa_n_sphere)
        )

        return results


def check_atom_consistency(u):
    """Checks if every frame in the trajectory has the same number of atoms."""
    n_topology = u.atoms.n_atoms
    inconsistent_frames = []

    print(f"Topology atom count: {n_topology}")

    for ts in u.trajectory:
        if u.atoms.n_atoms != n_topology:
            inconsistent_frames.append((ts.frame, u.atoms.n_atoms))

    if not inconsistent_frames:
        print("All frames are consistent.")
        return True
    else:
        print("Inconsistency detected!")
        for frame, count in inconsistent_frames:
            print(f"Frame {frame}: found {count} atoms (Expected {n_topology})")
        return False
