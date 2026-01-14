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


class ProteinAnalyzer:
    def __init__(self, pdb_path, xtc_path=None):
        """Initializes the Universe and checks the system size."""
        assert pdb_path.exists(), f"Expected file {pdb_path} to exist, but it does not."
        assert xtc_path.exists(), f"Expected file {xtc_path} to exist, but it does not."

        self.pdb_path = pdb_path
        self.xtc_path = xtc_path

        if isinstance(pdb_path, list):
            # Use the first PDB as the topology (structure definition)
            # Use the full list as the trajectory (the frames)
            topology = pdb_path[0]
            trajectory = pdb_path
        else:
            topology = pdb_path
            trajectory = xtc_path  # This can be None, a string, or a list

        if trajectory is None:
            self.md_analysis_u = MDAnalysis.Universe(topology)
            self.md_traj = md.load(self.pdb_path)
            raise RuntimeError(
                "Be carefull no trajectory is loaded. Delete this raise if you known what you do"
            )
        else:
            self.md_analysis_u = MDAnalysis.Universe(topology, trajectory)
            self.md_traj = md.load(self.xtc_path, top=self.pdb_path)

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

        results["avg_squared_Ree"], results["std_squared_Ree"] = mean_std(
            compute_end_to_end_distance(self.md_analysis_u, self.residues), squared=True
        )
        results["avg_maximum_diameter"], results["std_maximum_diameter"] = mean_std(
            compute_maximum_diameter(self.md_analysis_u), squared=False
        )
        gyration_output = compute_gyration_tensor_properties(self.md_traj)
        results.update(
            get_ensemble_summary(
                gyration_output, include_min_max=False, include_histogram=False, bins=20
            )
        )

        results["scaling_exponent"] = compute_scaling_exponent(
            self.md_traj, min_sep=scaling_min_sep
        )

        # results["chimical_shift"] = compute_chemical_shift(self.md_traj) # Need the conda env of LEGOLAS to be use -> So computed in another script afterward
        results["local_chirality"] = compute_local_chirality(self.md_traj)
        results.update(compute_secondary_structure_propensities(self.md_traj))

        results["dccm"] = compute_dccm(self.md_analysis_u, self.protein_atoms)
        results.update(pooled_dihedral_entropy(self.md_traj, bins=60))

        results["distance_fluctuations"] = compute_distance_fluctuations(
            self.md_analysis_u, self.protein_atoms
        )
        results["contact_map"] = compute_contact_map(
            self.md_analysis_u, self.protein_atoms, cutoff=contact_cutoff
        )

        results.update(
            compute_residue_sasa(self.md_traj, n_sphere_points=sasa_n_sphere)
        )

        return results
