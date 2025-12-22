import MDAnalysis
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dssp import DSSP
import numpy as np
import freesasa
from prody import Ensemble
from scipy.optimize import curve_fit
import mdtraj as md
from collections import defaultdict


class ProteinAnalyzer:
    def __init__(self, pdb_path, xtc_path):
        """Initializes the Universe and checks the system size."""
        self.pdb_path = pdb_path
        self.xtc_path = xtc_path
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
        start_ca = self.u.select_atoms(f"resid {self.residues[0].resid} and name CA")
        end_ca = self.u.select_atoms(f"resid {self.residues[-1].resid} and name CA")
        # Ensure single atom selections
        assert len(start_ca) == 1, "Start residue CA atom not found."
        assert len(end_ca) == 1, "End residue CA atom not found."

        all_distances = []
        for ts in self.u.trajectory:
            resA, resB, dist = distances.dist(start_ca, end_ca)
            all_distances.append(dist)

        return np.array(all_distances)

    def compute_radius_of_gyration(self):
        """Calculates the Radius of Gyration (Rg) over time."""
        rg_values = []
        for ts in self.u.trajectory:
            rg_values.append(self.protein_atoms.radius_of_gyration())
        return np.array(rg_values)

    def compute_mean_squared_end_to_end_distance(self):
        """Computes the Mean Squared End-to-End Distance (MSED) over the trajectory."""
        dists = self.compute_end_to_end_distance()
        return np.mean(dists**2)

    def compute_mean_squared_radius_of_gyration(self):
        """Computes the Mean Squared Radius of Gyration (MSRG) over the trajectory."""
        rg_values = self.compute_radius_of_gyration()
        return np.mean(rg_values**2)

    def compute_secondary_structure_propensities(self):
        """
        Computes the propensity of each secondary structure element
        per residue over the trajectory.
        """
        # 1. Run the DSSP analysis
        # selection=self.u.select_atoms("protein") is recommended if you have ligands/solvent
        dssp_ana = DSSP(self.u).run()

        # 2. Use the one-hot encoded array (n_frames, n_residues, 3)
        # Index 0: Coil '-', Index 1: Helix 'H', Index 2: Sheet 'E'
        data = dssp_ana.results.dssp_ndarray

        # 3. Average across the frame axis (axis 0)
        # Resulting shape: (n_residues, 3)
        propensities = np.mean(data, axis=0)

        dict_propensities = {
            "residue": [res.resid for res in self.residues],
            "coil_propensity": propensities[:, 0],
            "helix_propensity": propensities[:, 1],
            "sheet_propensity": propensities[:, 2],
        }

        return dict_propensities

    # --- GLOBAL TOPOLOGY ---

    def compute_gyration_tensor_properties(self):
        results = {"asphericity": [], "prolateness": [], "eigenvalues": []}

        weights = self.protein_atoms.masses
        total_mass = self.protein_atoms.total_mass()

        for ts in self.u.trajectory:
            # 1. Manually compute the Gyration Tensor (Mass-weighted)
            pos = self.protein_atoms.positions - self.protein_atoms.center_of_mass()
            tensor = np.dot(pos.T, pos * weights[:, np.newaxis]) / total_mass

            eigvals = np.sort(np.linalg.eigvalsh(tensor))

            l1, l2, l3 = eigvals
            Rg2 = l1 + l2 + l3
            b = l3 - 0.5 * (l1 + l2)
            c = l2 - l1
            kappa2 = (b**2 + 0.75 * c**2) / (Rg2**2)  # or use MDTraj form below

            results["eigenvalues"].append(eigvals)
            results["asphericity"].append(b)
            results["prolateness"].append(kappa2)
        return results

    def compute_scaling_exponent(self, min_sep=5):
        """Fits ⟨R_ij⟩ ~ |i-j|^nu using C-alpha distances."""
        ca = self.protein_atoms.select_atoms("name CA")
        n_res = len(ca)

        acc = defaultdict(list)

        # loop over trajectory
        for ts in self.u.trajectory:
            coords = ca.positions
            for i in range(n_res):
                ri = coords[i]
                for j in range(i + 1, n_res):
                    n = j - i
                    r = np.linalg.norm(coords[j] - ri)
                    acc[n].append(r)

        # mean R(n)
        n_list = np.array(sorted(acc.keys()))
        R_mean = np.array([np.mean(acc[n]) for n in n_list])

        # log–log fit
        mask = n_list >= min_sep
        logn = np.log(n_list[mask])
        logR = np.log(R_mean[mask])

        p = np.polyfit(logn, logR, 1)
        nu = p[0]

        return nu

    # --- BACKBONE GRAMMAR ---

    def compute_dihedral_distribution_and_entropy(self):
        """Uses MDTraj to compute Dihedrals and X-Entropy for S_conf."""
        t = md.load(self.xtc_path, top=self.pdb_path)
        phi_indices, phi_angles = md.compute_phi(t)
        psi_indices, psi_angles = md.compute_psi(t)

        # Simple Shannon Entropy for the (phi, psi) distribution
        # In a real scenario, use 'X-Entropy' library for refined integration
        def calculate_entropy(angles):
            hist_counts, bin_edges = np.histogram(angles, bins=50, density=False)
            p = hist_counts / hist_counts.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            return entropy

        s_conf_phi = [
            calculate_entropy(phi_angles[:, i]) for i in range(phi_angles.shape[1])
        ]

        return {"phi": phi_angles, "psi": psi_angles, "entropy_phi": s_conf_phi}

    # --- NETWORK DYNAMICS ---

    def compute_dccm(self):
        """
        Calculates the DCCM from an ensemble by first aligning to the mean.
        Uses the direct covariance calculation to avoid ProDy class-type conflicts.
        """
        # 1. Coordinate Extraction
        ca = self.protein_atoms.select_atoms("name CA")
        n_frames = len(self.u.trajectory)
        n_atoms = len(ca)

        coords = np.zeros((n_frames, n_atoms, 3))
        for i, ts in enumerate(self.u.trajectory):
            coords[i] = ca.positions

        # 2. Build the Ensemble and Align to Mean
        ensemble = Ensemble("Protein")
        ensemble.setCoords(coords[0])
        ensemble.addCoordset(coords)

        # Aligning is crucial to remove global rotation/translation
        try:
            # ProDy's iterative superposition finds the average structure
            from prody.measure import iterativeSuperpose

            iterativeSuperpose(ensemble)
        except ImportError:
            # Fallback for version/import issues
            from prody.measure import superpose

            superpose(ensemble, ensemble)

        # 3. FIX: Calculate Correlation directly from the aligned coordinates
        # This avoids the "TypeError: modes must be a Mode..." error
        # because we are passing the numerical coordinate sets.
        aligned_coords = ensemble.getCoordsets()  # Shape (n_frames, n_atoms, 3)

        # Calculate mean structure
        mean_coords = aligned_coords.mean(axis=0)

        # Calculate fluctuations (ΔR = R - <R>)
        fluctuations = aligned_coords - mean_coords

        # Compute the dot products for the cross-correlation matrix
        # C_ij = <ΔRi · ΔRj> / sqrt(<ΔRi^2> * <ΔRj^2>)
        dccm = np.zeros((n_atoms, n_atoms))

        # Vectorized variance calculation for normalization
        # variances shape: (n_atoms,)
        variances = np.mean(np.sum(fluctuations**2, axis=2), axis=0)
        norm_factor = np.sqrt(variances) + 1e-12  # Avoid division by zero

        for i in range(n_atoms):
            # Calculate dot product across all frames for atom i and all atoms j
            # (n_frames, 3) dot (n_frames, n_atoms, 3)
            dot_products = np.mean(
                np.sum(fluctuations[:, i : i + 1, :] * fluctuations, axis=2), axis=0
            )
            dccm[i, :] = dot_products / (norm_factor[i] * norm_factor)

        return dccm

    def compute_distance_fluctuations(self):
        """Identifies rigid vs flexible pairs via SD of distance matrix."""
        ca = self.protein_atoms.select_atoms("name CA")
        n_ca = len(ca)
        sum_sq_dist = np.zeros((n_ca, n_ca))
        sum_dist = np.zeros((n_ca, n_ca))
        n_frames = len(self.u.trajectory)

        for ts in self.u.trajectory:
            d = distances.distance_array(ca.positions, ca.positions)
            sum_dist += d
            sum_sq_dist += d**2

        # Variance = E[X^2] - (E[X])^2
        mean_dist = sum_dist / n_frames
        variance = (sum_sq_dist / n_frames) - (mean_dist**2)
        return np.sqrt(np.maximum(variance, 0))  # Return Standard Deviation

    # --- GEOMETRIC FEATURES ---

    def compute_contact_map(self, cutoff=8.0):
        """Calculates average contact frequency matrix."""
        ca = self.protein_atoms.select_atoms("name CA")
        n_res = len(ca)
        contact_sum = np.zeros((n_res, n_res))

        for ts in self.u.trajectory:
            d = distances.distance_array(ca.positions, ca.positions)
            contact_sum += (d < cutoff).astype(int)

        return contact_sum / len(self.u.trajectory)

    # --- SOLVENT-SOLUTE ---

    def compute_trajectory_sasa_mdtraj(self):
        """Calculates SASA for every frame using MDTraj (Shrake-Rupley)."""
        # Load trajectory into mdtraj format
        t = md.load(self.xtc_path, top=self.pdb_path)

        sasa_per_residue = md.shrake_rupley(
            t, mode="residue"
        )  # shape (n_frames, n_residues)
        total_sasa = sasa_per_residue.sum(axis=1)

        return total_sasa  # Returns np.array of shape (n_frames,)

    def compute_hydration_density(self, bins=50, Rmax=30.0):
        """
        Compute time-averaged 3D hydration density of solvent around the protein
        in a protein-centered reference frame.
        """
        solvent = self.u.select_atoms("resname SOL or resname WAT or resname HOH")

        n_frames = len(self.u.trajectory)

        # Define fixed spatial domain (protein-centered)
        my_range = [[-Rmax, Rmax], [-Rmax, Rmax], [-Rmax, Rmax]]

        # Use float array: density is real-valued
        grid = np.zeros((bins, bins, bins), dtype=float)

        for ts in self.u.trajectory:
            protein_com = self.protein_atoms.center_of_mass()
            relative_pos = solvent.positions - protein_com

            h, _ = np.histogramdd(relative_pos, bins=bins, range=my_range)
            grid += h

        # Time average
        grid /= n_frames

        return grid
