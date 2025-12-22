import MDAnalysis
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dssp import DSSP
import numpy as np
import freesasa
from prody import Ensemble
from scipy.optimize import curve_fit
import mdtraj as md


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

        all_distances = []
        for ts in self.u.trajectory:
            start_ca = self.u.select_atoms(
                f"resid {self.residues[0].resid} and name CA"
            )
            end_ca = self.u.select_atoms(f"resid {self.residues[-1].resid} and name CA")
            resA, resB, dist = distances.dist(start_ca, end_ca)
            all_distances.append(dist**2)

        return np.array(all_distances)

    def compute_radius_of_gyration(self):
        """Calculates the Radius of Gyration (Rg) over time."""
        rg_values = []
        for ts in self.u.trajectory:
            rg = self.protein_atoms.atoms.radius_of_gyration()
            rg_values.append(rg)

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
        """Computes Eigenvalues, Asphericity (b), and Prolateness (S)."""
        results = {"asphericity": [], "prolateness": [], "eigenvalues": []}

        for ts in self.u.trajectory:
            # Get the 3x3 gyration tensor
            tensor = (
                self.protein_atoms.moment_of_inertia() / self.protein_atoms.total_mass()
            )
            eigvals = np.sort(np.linalg.eigvalsh(tensor))  # lambda1 < lambda2 < lambda3
            l1, l2, l3 = eigvals

            # Asphericity (b)
            b = l3 - 0.5 * (l1 + l2)

            # Prolateness (S) - Shape descriptor
            # Using the standard definition relative to the mean eigenvalue
            mean_eig = np.mean(eigvals)
            s = ((l1 - mean_eig) * (l2 - mean_eig) * (l3 - mean_eig)) / (mean_eig**3)

            results["eigenvalues"].append(eigvals)
            results["asphericity"].append(b)
            results["prolateness"].append(s)

        return results

    def compute_scaling_exponent(self):
        """Fits internal distances R_ij to |i-j|^nu to find scaling exponent."""
        # Calculate mean distance between all pairs of C-alpha atoms
        ca = self.protein_atoms.select_atoms("name CA")
        n_res = len(ca)
        dist_matrix = np.zeros((n_res, n_res))

        # Average distance matrix over trajectory
        for ts in self.u.trajectory:
            dist_matrix += distances.distance_array(ca.positions, ca.positions)
        dist_matrix /= len(self.u.trajectory)

        # Extract internal distances |i-j| vs R_ij
        separations = []
        actual_dists = []
        for i in range(n_res):
            for j in range(i + 1, n_res):
                separations.append(abs(i - j))
                actual_dists.append(dist_matrix[i, j])

        # Power law fit: R = A * N^nu
        def power_law(n, a, nu):
            return a * n**nu

        popt, _ = curve_fit(power_law, separations, actual_dists, p0=[3.8, 0.5])
        return popt[1]  # Return nu

    # --- BACKBONE GRAMMAR ---

    def compute_dihedral_distribution_and_entropy(self):
        """Uses MDTraj to compute Dihedrals and X-Entropy for S_conf."""
        t = md.load(self.xtc_path, top=self.pdb_path)
        phi_indices, phi_angles = md.compute_phi(t)
        psi_indices, psi_angles = md.compute_psi(t)

        # Simple Shannon Entropy for the (phi, psi) distribution
        # In a real scenario, use 'X-Entropy' library for refined integration
        def calculate_entropy(angles):
            hist, _ = np.histogram(angles, bins=50, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))

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
        norm_factor = np.sqrt(variances)

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

    def compute_exact_sasa(self):
        """Numerical integration of SASA using FreeSASA."""
        # FreeSASA works best with Bio.PDB or structure files
        struct = freesasa.Structure(self.pdb_path)
        result = freesasa.calc(struct)
        return result.totalArea()

    def compute_hydration_density(self, bins=50):
        """3D Volumetric histogramming of solvent (Water) around protein."""
        solvent = self.u.select_atoms("resname SOL or resname WAT")
        # Centering the protein for consistent grid mapping
        all_positions = []
        for ts in self.u.trajectory:
            protein_com = self.protein_atoms.center_of_mass()
            relative_pos = solvent.positions - protein_com
            all_positions.append(relative_pos)

        all_positions = np.concatenate(all_positions)
        grid, edges = np.histogramdd(all_positions, bins=bins)
        return grid
