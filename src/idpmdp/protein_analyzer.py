import MDAnalysis
from MDAnalysis.analysis import distances
import numpy as np
from prody import Ensemble
import mdtraj as md
from idpmdp.utils import get_ensemble_summary, mean_std
from scipy.stats import linregress
import h5py
from pathlib import Path
from scipy.spatial.distance import pdist
from prody.measure import superpose


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

    def compute_end_to_end_distance(self):
        """
        Calculates the distance between the CA atoms of the
        first and last residues across the trajectory.
        """
        start_ca = self.md_analysis_u.select_atoms(
            f"resid {self.residues[0].resid} and name CA"
        )
        end_ca = self.md_analysis_u.select_atoms(
            f"resid {self.residues[-1].resid} and name CA"
        )
        # Ensure single atom selections
        assert (
            len(start_ca) == 1
        ), f"Start residue CA atom not found. Available atoms names: {self.residues[0].atoms.names}"
        assert (
            len(end_ca) == 1
        ), f"End residue CA atom not found.Available atoms names: {self.residues[0].atoms.names}"

        all_distances = []
        for ts in self.md_analysis_u.trajectory:
            resA, resB, dist = distances.dist(start_ca, end_ca)
            all_distances.append(dist)

        return np.array(all_distances)

    def compute_maximum_diameter(self):
        """
        Calculates the maximum distance between any two CA atoms
        (Dmax) for each frame in the trajectory.
        """
        # 1. Pre-select CA atoms to avoid overhead inside the loop
        ca_atoms = self.md_analysis_u.select_atoms("name CA")

        all_dmax = []

        # 2. Iterate through trajectory
        for ts in self.md_analysis_u.trajectory:
            # Get the coordinates for the current frame
            coords = ca_atoms.positions

            # pdist computes the distance between every pair of atoms (n*(n-1)/2 distances)
            # np.max gives us the largest of those distances
            dmax = np.max(pdist(coords))
            all_dmax.append(dmax)

        return np.array(all_dmax)

    def compute_secondary_structure_propensities(self):
        """
        Computes the SS8 (DSSP) propensity of each secondary structure element
        per residue over the trajectory using MDTraj.
        """

        # 1. Load trajectory with MDTraj
        traj = md.load_xtc(self.xtc_path, top=self.pdb_path)

        # Optional: restrict to protein residues only
        traj = traj.atom_slice(traj.top.select("protein"))

        # 2. Compute DSSP with full 8-class output
        # Output shape: (n_frames, n_residues)
        dssp = md.compute_dssp(traj, simplified=False)

        # 3. Map DSSP symbols to SS8 labels (space -> C)
        mapping = {
            "H": "H",  # alpha helix
            "G": "G",  # 3-10 helix
            "I": "I",  # pi helix
            "E": "E",  # beta strand
            "B": "B",  # beta bridge
            "T": "T",  # turn
            "S": "S",  # bend
            " ": "C",  # coil
        }

        ss8_labels = ["H", "G", "I", "E", "B", "T", "S", "C"]
        n_frames, n_residues = dssp.shape

        # 4. Compute per-residue propensities
        propensities = {k: np.zeros(n_residues) for k in ss8_labels}

        for f in range(n_frames):
            for i, code in enumerate(dssp[f]):
                propensities[mapping[code]][i] += 1

        # Normalize by number of frames
        for k in propensities:
            propensities[k] /= n_frames

        # 5. Sanity check on residue indexing (as in your original code)
        resids = np.array([res.resSeq for res in traj.topology.residues])
        assert np.all(
            np.diff(resids) > 0
        ), "Residues are not in order or contain duplicates!"
        assert np.all(
            np.diff(resids) == 1
        ), f"Gap detected in residue sequence: {resids}"

        return {f"ss_propensity_{k}": v for k, v in propensities.items()}

    def compute_max_diameter(self):
        # Get coordinates of C-alpha atoms for the current frame
        ca_coords = self.md_analysis_u.select_atoms("name CA").positions
        # pdist computes pairwise distances between all atoms
        distances = pdist(ca_coords)
        return np.max(distances)

    # --- GLOBAL TOPOLOGY ---

    def compute_gyration_tensor_properties(self):
        results = {
            "gyration_eigenvalues_l1": [],
            "gyration_eigenvalues_l2": [],
            "gyration_eigenvalues_l3": [],
            "gyration_l1/l2": [],
            "gyration_l1/l3": [],
            "gyration_l2/l3": [],
            "radius_of_gyration": [],
            "asphericity": [],
            "normalized_acylindricity": [],
            "rel_shape_anisotropy": [],
            "prolateness": [],
        }

        protein_traj = self.md_traj.atom_slice(self.md_traj.topology.select("protein"))

        eigvals = md.principal_moments(protein_traj)

        for one_set_of_eigvals in eigvals:
            l1, l2, l3 = one_set_of_eigvals
            results["gyration_eigenvalues_l1"].append(l1)
            results["gyration_eigenvalues_l2"].append(l2)
            results["gyration_eigenvalues_l3"].append(l3)
            results["gyration_l1/l2"].append(l1 / l2 if l2 > 0 else 0)
            results["gyration_l1/l3"].append(l1 / l3 if l3 > 0 else 0)
            results["gyration_l2/l3"].append(l2 / l3 if l3 > 0 else 0)

            rg2 = (
                l1 + l2 + l3
            )  # The square of the radius of gyration and also the trace of the matrix
            rg = np.sqrt(rg2)
            results["radius_of_gyration"].append(rg)

            asphericity = ((l1 - l2) ** 2 + (l1 - l3) ** 2 + (l2 - l3) ** 2) / (
                2 * (rg2**2)
            )
            results["asphericity"].append(asphericity)

            numerator_p = (2 * l1 - l2 - l3) * (2 * l2 - l1 - l3) * (2 * l3 - l1 - l2)
            denominator_p = (
                2 * (l1**2 + l2**2 + l3**2 - l1 * l2 - l1 * l3 - l2 * l3) ** 1.5
            )
            prolateness = numerator_p / denominator_p
            results["prolateness"].append(prolateness)

            normalized_acylindricity = (l2 - l3) / rg2
            results["normalized_acylindricity"].append(normalized_acylindricity)

            # Assuming l1, l2, l3 are predefined numeric values
            numerator_kappa = 3 * (l1 * l2 + l2 * l3 + l3 * l1)
            denominator_kappa = (l1 + l2 + l3) ** 2
            rel_shape_anisotropy = 1 - (numerator_kappa / denominator_kappa)
            results["rel_shape_anisotropy"].append(rel_shape_anisotropy)

        return {key: np.array(value) for key, value in results.items()}

    def compute_scaling_exponent(self, min_sep=5):
        """Scaling exponent ν using self.md_traj - Cα selection inside."""

        # Select only Cα atoms
        ca_mask = self.md_traj.topology.select("name CA")
        traj_ca = self.md_traj.atom_slice(ca_mask)
        n_res = len(ca_mask)

        # Generate all Cα pairs with separation >= min_sep
        pairs = []
        for i in range(n_res):
            for j in range(i + min_sep, n_res):
                pairs.append([i, j])
        pairs = np.array(pairs)

        # Compute distances and bin by separation
        dists = md.compute_distances(traj_ca, pairs)
        r_mean = dists.mean(axis=0)
        separations = pairs[:, 1] - pairs[:, 0]

        unique_seps = np.arange(min_sep, n_res)
        r_binned = np.array(
            [
                r_mean[separations == s].mean()
                for s in unique_seps
                if np.any(separations == s)
            ]
        )

        # Log-log fit: log(r) = ν*log(s) + c
        valid = np.isfinite(r_binned) & (r_binned > 0)
        log_s = np.log(unique_seps[valid][: len(r_binned[valid])])
        log_r = np.log(r_binned[valid])

        nu = linregress(log_s, log_r).slope

        return nu

    def compute_dihedral_distribution(self):
        """Uses MDTraj to compute Dihedrals and X-Entropy for S_conf."""

        phi_indices, phi_angles = md.compute_phi(self.md_traj)
        psi_indices, psi_angles = md.compute_psi(self.md_traj)

        n_frames = self.md_traj.n_frames
        n_residues = self.md_traj.topology.n_residues  # Total length L

        # 2. Initialize full-sized arrays filled with NaNs
        # Shape: (n_frames, n_residues)
        full_phi = np.full((n_frames, n_residues), np.nan)
        full_psi = np.full((n_frames, n_residues), np.nan)

        # 3. Place the computed angles into the correct positions
        # Phi starts at residue 1 (skips 0)
        full_phi[:, 1:] = phi_angles

        # Psi ends at residue L-2 (skips L-1)
        full_psi[:, :-1] = psi_angles

        return {"phi": full_phi, "psi": full_psi}

    def compute_dccm(self):
        ca = self.protein_atoms.select_atoms("name CA")
        n_frames = len(self.md_analysis_u.trajectory)
        n_atoms = len(ca)

        # 1. Coordinate Extraction
        coords = np.zeros((n_frames, n_atoms, 3))
        for i, ts in enumerate(self.md_analysis_u.trajectory):
            coords[i] = ca.positions

        # 2. Alignment via ProDy
        ensemble = Ensemble("Protein")
        ensemble.setCoords(coords[0])  # Reference is frame 0
        ensemble.addCoordset(coords)
        superpose(ensemble, ensemble)
        aligned_coords = ensemble.getCoordsets()

        # 3. Fluctuation Calculation
        mean_coords = aligned_coords.mean(axis=0)
        fluctuations = aligned_coords - mean_coords

        # 4. Vectorized DCCM (The Fast Way)
        # Resulting shape: (n_atoms, n_atoms)
        dot_products = np.einsum("fai,fbi->ab", fluctuations, fluctuations) / n_frames

        # 5. Normalization (C_ij = <ΔRi·ΔRj> / sqrt(<ΔRi²><ΔRj²>))
        variances = np.diag(dot_products)
        # Using outer product to create the denominator matrix
        norm_matrix = np.sqrt(np.outer(variances, variances)) + 1e-12
        dccm = dot_products / norm_matrix

        return dccm

    def compute_distance_fluctuations(self):
        """Identifies rigid vs flexible pairs via SD of distance matrix."""
        ca = self.protein_atoms.select_atoms("name CA")
        n_ca = len(ca)
        sum_sq_dist = np.zeros((n_ca, n_ca))
        sum_dist = np.zeros((n_ca, n_ca))
        n_frames = len(self.md_analysis_u.trajectory)

        for ts in self.md_analysis_u.trajectory:
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

        for ts in self.md_analysis_u.trajectory:
            d = distances.distance_array(ca.positions, ca.positions)
            contact_sum += (d < cutoff).astype(int)

        return contact_sum / len(self.md_analysis_u.trajectory)

    # --- SOLVENT-SOLUTE ---

    def compute_residue_sasa(self, n_sphere_points, stride):
        """
        Returns a 1D numpy array of time-averaged SASA values
        in the order of the protein sequence.
        """
        # MaxASA values in nm^2 (Tien et al. 2013)
        TIEN_MAX_SASA = {
            "ALA": 1.21,
            "ARG": 2.48,
            "ASN": 1.87,
            "ASP": 1.87,
            "CYS": 1.48,
            "GLN": 2.14,
            "GLU": 2.14,
            "GLY": 0.97,
            "HIS": 2.16,
            "ILE": 1.95,
            "LEU": 1.91,
            "LYS": 2.30,
            "MET": 2.03,
            "PHE": 2.28,
            "PRO": 1.54,
            "SER": 1.43,
            "THR": 1.63,
            "TRP": 2.64,
            "TYR": 2.55,
            "VAL": 1.65,
        }

        # Load the trajectory
        t = md.load(self.xtc_path, top=self.pdb_path, stride=stride)

        # Select only the protein (standard practice to avoid membrane/solvent interference)
        protein_indices = t.topology.select("protein")
        t_prot = t.atom_slice(protein_indices)

        # Compute SASA per residue
        # Returns shape (n_frames, n_residues)
        sasa_per_frame_per_res = md.shrake_rupley(
            t_prot, n_sphere_points=n_sphere_points, mode="residue"
        )
        # Calculate Mean (Average) and Std (Fluctuations) for both metrics
        # Absolute SASA (nm^2)
        avg_abs_sasa = np.mean(sasa_per_frame_per_res, axis=0)
        std_abs_sasa = np.std(sasa_per_frame_per_res, axis=0)

        # Compute Relative Solvent Accessibility (RSA)
        residue_names = [res.name[:3].upper() for res in t_prot.topology.residues]
        max_vals = np.array([TIEN_MAX_SASA.get(name, 1.0) for name in residue_names])

        # This divides every frame's SASA by the MaxSASA for that residue type
        rsa_per_frame = sasa_per_frame_per_res / max_vals

        # Relative SASA (RSA - normalized 0 to 1)
        avg_rel_sasa = np.mean(rsa_per_frame, axis=0)
        std_rel_sasa = np.std(rsa_per_frame, axis=0)

        return {
            "sasa_abs_mean": avg_abs_sasa,  # Physical area (nm^2)
            "sasa_abs_std": std_abs_sasa,  # Area fluctuations
            "sasa_rel_mean": avg_rel_sasa,  # Normalized exposure (0-1)
            "sasa_rel_std": std_rel_sasa,  # Relative flexibility
        }

    def pooled_dihedral_entropy(self, bins=60):
        """
        phi_array: shape (n_frames, n_residues)
        psi_array: shape (n_frames, n_residues)
        """
        results = self.compute_dihedral_distribution()
        phi_array = results["phi"]
        psi_array = results["psi"]

        n_frames, n_res = phi_array.shape
        pooled_entropy = {"phi_dihedrals_entropy": [], "psi_dihedrals_entropy": []}

        for i in range(n_res):
            # Process PHI for residue i
            phi_counts, _ = np.histogram(
                phi_array[:, i], bins=bins, range=[-np.pi, np.pi]
            )
            p_phi = phi_counts / n_frames
            p_phi = p_phi[p_phi > 0]  # Remove zeros to avoid log(0)
            s_phi = -np.sum(p_phi * np.log(p_phi))

            # Process PSI for residue i
            psi_counts, _ = np.histogram(
                psi_array[:, i], bins=bins, range=[-np.pi, np.pi]
            )
            p_psi = psi_counts / n_frames
            p_psi = p_psi[p_psi > 0]
            s_psi = -np.sum(p_psi * np.log(p_psi))

            pooled_entropy["phi_dihedrals_entropy"].append(s_phi)
            pooled_entropy["psi_dihedrals_entropy"].append(s_psi)

        pooled_entropy["phi_dihedrals_entropy"] = np.array(
            pooled_entropy["phi_dihedrals_entropy"]
        )
        pooled_entropy["psi_dihedrals_entropy"] = np.array(
            pooled_entropy["psi_dihedrals_entropy"]
        )
        return pooled_entropy

    def compute_all(
        self,
        sasa_n_sphere=960,
        sasa_stride=1,
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
            self.compute_end_to_end_distance(), squared=True
        )
        results["avg_maximum_diameter"], results["std_maximum_diameter"] = mean_std(
            self.compute_maximum_diameter(), squared=False
        )
        gyration_output = self.compute_gyration_tensor_properties()
        results.update(
            get_ensemble_summary(
                gyration_output, include_min_max=False, include_histogram=False, bins=20
            )
        )

        results["scaling_exponent"] = self.compute_scaling_exponent(
            min_sep=scaling_min_sep
        )
        results.update(self.compute_secondary_structure_propensities())
        results.update(self.pooled_dihedral_entropy())
        results["dccm"] = self.compute_dccm()
        results["distance_fluctuations"] = self.compute_distance_fluctuations()
        results["contact_map"] = self.compute_contact_map(cutoff=contact_cutoff)
        results.update(
            self.compute_residue_sasa(n_sphere_points=sasa_n_sphere, stride=sasa_stride)
        )

        return results

    def save_all(self, results, output_folder: Path):
        filename = output_folder / "properties.h5"

        with h5py.File(filename, "w") as hf:
            for key, value in results.items():
                # Convert lists/tuples to numpy arrays for HDF5 compatibility
                if isinstance(value, (list, tuple)):
                    value = np.array(value)

                if isinstance(value, np.ndarray) and value.ndim > 0:
                    hf.create_dataset(
                        key,
                        data=value,
                        dtype="float32",
                        compression="gzip",
                        compression_opts=4,  # Ranges from 0-9
                    )
                else:
                    # Scalars cannot be compressed in HDF5
                    hf.create_dataset(key, data=value)

        print(f"Successfully saved data to {filename}")
