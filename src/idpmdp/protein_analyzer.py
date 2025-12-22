import MDAnalysis
from MDAnalysis.analysis import distances
import numpy as np


class ProteinAnalyzer:
    def __init__(self, pdb_path, xtc_path):
        """Initializes the Universe and checks the system size."""
        self.u = MDAnalysis.Universe(pdb_path, xtc_path)
        # Verify that the universe contains only one segment
        assert (
            len(self.u.segments) == 1
        ), "The provided PDB file contains multiple segments."

        # Identify the protein and its size
        self.protein_atoms = self.u.select_atoms("protein")
        self.residues = self.protein_atoms.residues
        self.protein_size = len(self.residues)

        print(f"Loaded Universe with {len(self.u.trajectory)} frames.")
        print(f"Protein size: {self.protein_size} residues.")

    def compute_end_to_end_distance(self):
        """
        Calculates the distance between the CA atoms of the
        first and last residues across the trajectory.
        """

        all_distances = []
        for ts in self.u.trajectory:
            start_ca = self.u.select_atoms(
                f"resid {self.residues[0].resid} and name CA"
            )
            end_ca = self.u.select_atoms(f"resid {self.residues[-1].resid} and name CA")
            dist = distances(start_ca.positions, end_ca.positions)
            assert dist.shape == (
                1,
                1,
            ), "Distance calculation did not return expected shape."
            dist = dist[0][0]
            all_distances.append(dist**2)

        return np.array(all_distances)

    def compute_radius_of_gyration(self):
        """Calculates the Radius of Gyration (Rg) over time."""
        rg_values = []
        for ts in self.u.trajectory:
            rg = self.protein_atoms.atoms.radius_of_gyration()
            rg_values.append(rg)

        return np.array(rg_values)

    def compute_mean_squared_distance(self):
        """Computes the MSD specifically for the end-to-end vector."""
        dists = self.compute_end_to_end_distance()
        return np.mean(dists**2)
