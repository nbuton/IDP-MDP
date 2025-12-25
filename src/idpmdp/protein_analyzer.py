import MDAnalysis
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dssp import DSSP
import numpy as np
from prody import Ensemble
import mdtraj as md
import logging


class ProteinAnalyzer:
    def __init__(self, pdb_path, xtc_path=None):
        """Initializes the Universe and checks the system size."""
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
            self.u = MDAnalysis.Universe(topology)
        else:
            self.u = MDAnalysis.Universe(topology, trajectory)
        # Verify that the universe contains only one segment
        assert (
            len(self.u.segments) == 1
        ), "The provided PDB file contains multiple segments."

        # Identify the protein and its size
        all_atom_backbone = self.u.select_atoms("name N CA C O")
        self.is_coarse_grained = len(all_atom_backbone) < (len(self.u.residues) * 4)
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
            radius_of_gyration = self.protein_atoms.radius_of_gyration()
            rg_values.append(radius_of_gyration)
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
        if self.is_coarse_grained:
            backbone = self.u.select_atoms("name N or name CA or name C or name O")
            dssp_ana = DSSP(backbone).run(guess_hydrogens=True)
        else:
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
        """Vectorized calculation of the scaling exponent."""
        ca = self.protein_atoms.select_atoms("name CA")
        n_res = len(ca)
        n_frames = len(self.u.trajectory)

        # Pre-allocate an array for the sum of distance matrices
        # self_distance_array returns the upper triangle as a 1D vector
        dist_vec_sum = np.zeros(int(n_res * (n_res - 1) / 2))

        # 1. Optimized trajectory loop (C-level distance calculation)
        for ts in self.u.trajectory:
            dist_vec_sum += distances.self_distance_array(ca.positions)

        # Mean distance vector across trajectory
        dist_vec_avg = dist_vec_sum / n_frames

        # 2. Map the 1D distance vector to separations |i-j|
        # Instead of loops, we use the indices of the upper triangle
        iu = np.triu_indices(n_res, k=1)
        separations = iu[1] - iu[0]  # This gives us |i-j| for every entry in dist_vec

        # 3. Vectorized mean per separation
        n_list = np.arange(min_sep, n_res)
        r_mean = []

        for n in n_list:
            # Mask the distance vector where separation is exactly n
            r_mean.append(np.mean(dist_vec_avg[separations == n]))

        r_mean = np.array(r_mean)

        # 4. Log-Log fit
        logn = np.log(n_list)
        logR = np.log(r_mean)

        nu, _ = np.polyfit(logn, logR, 1)
        return nu

    # --- BACKBONE GRAMMAR ---

    def compute_dihedral_distribution_and_entropy(self):
        """Uses MDTraj to compute Dihedrals and X-Entropy for S_conf."""
        if self.xtc_path is None:
            t = md.load(self.pdb_path)
        else:
            t = md.load(self.xtc_path, top=self.pdb_path)
        phi_indices, phi_angles = md.compute_phi(t)
        psi_indices, psi_angles = md.compute_psi(t)
        return {"phi": phi_angles, "psi": psi_angles}

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

    def compute_residue_sasa(self, n_sphere_points, stride):
        """
        Returns a 1D numpy array of time-averaged SASA values
        in the order of the protein sequence.
        """
        # 1. Load the trajectory
        if self.xtc_path is None:
            t = md.load(self.pdb_path, stride=stride)
        else:
            t = md.load(self.xtc_path, top=self.pdb_path, stride=stride)

        # 2. Select only the protein (standard practice to avoid membrane/solvent interference)
        protein_indices = t.topology.select("protein")
        t_prot = t.atom_slice(protein_indices)

        # 3. Compute SASA per residue
        # Returns shape (n_frames, n_residues)
        sasa_per_frame_per_res = md.shrake_rupley(t_prot, mode="residue")

        # 4. Average over time (frames)
        # Resulting shape: (n_residues,)
        avg_sasa_array = np.mean(sasa_per_frame_per_res, axis=0)

        return avg_sasa_array

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

    def pooled_gyration_properties(self):
        """
        Pools the gyration tensor properties into frame-independent descriptors.
        """

        results_dict = self.compute_gyration_tensor_properties()
        asphericity = np.array(results_dict["asphericity"])
        prolateness = np.array(results_dict["prolateness"])
        eigenvals = np.array(results_dict["eigenvalues"])  # Shape (T, 3)

        pooled = {
            # Mean values (First Moment)
            "mean_asphericity": np.mean(asphericity),
            "mean_prolateness": np.mean(prolateness),
            "mean_eigenvalues": np.mean(eigenvals, axis=0),
            # Fluctuations (Second Moment)
            "std_asphericity": np.std(asphericity),
            # Distribution (For Histograms)
            "asphericity_hist": np.histogram(asphericity, bins=30, density=True),
            "prolateness_hist": np.histogram(prolateness, bins=30, density=True),
        }
        return pooled

    def pooled_dihedral_entropy(self, bins=60):
        """
        phi_array: shape (n_frames, n_residues)
        psi_array: shape (n_frames, n_residues)
        """
        results = self.compute_dihedral_distribution_and_entropy()
        phi_array = results["phi"]
        psi_array = results["psi"]

        n_frames, n_res = phi_array.shape
        pooled_entropy = {"phi": [], "psi": []}

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

            pooled_entropy["phi"].append(s_phi)
            pooled_entropy["psi"].append(s_psi)

        return pooled_entropy

    def hydration_density_on_sequence(self, bins, Rmax, cutoff):
        """
        Maps 3D hydration density back to a 1D sequence array.
        """
        grid = self.compute_hydration_density(bins=bins, Rmax=Rmax)
        # 1. Reconstruct the grid coordinate system
        lin = np.linspace(-Rmax, Rmax, bins)
        grid_x, grid_y, grid_z = np.meshgrid(lin, lin, lin, indexing="ij")
        # Create a (N_voxels, 3) array of all grid point coordinates
        grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        grid_flat = grid.ravel()

        n_residues = len(self.protein_atoms.residues)
        hydration_per_res = np.zeros(n_residues)

        # 2. Get protein COM to match the centering used in your grid calculation
        protein_com = self.protein_atoms.center_of_mass()

        # 3. For each residue, find which grid points are nearby
        for i, res in enumerate(self.protein_atoms.residues):
            # Shift residue positions to protein-centered frame
            res_pos = res.atoms.positions - protein_com

            # We use a KDTree or simple distance check to find voxels near the residue
            # For simplicity in this snippet, we'll check distances:
            # (For large grids, using scipy.spatial.cKDTree is much faster)
            for atom_pos in res_pos:
                dist_sq = np.sum((grid_coords - atom_pos) ** 2, axis=1)
                mask = dist_sq < cutoff**2

                # Sum the density in these voxels
                hydration_per_res[i] += np.sum(grid_flat[mask])

        # 4. Return normalized sequence array
        return hydration_per_res

    def compute_all(
        self,
        sasa_n_sphere=100,
        sasa_stride=1,
        hydration_bins=50,
        hydration_rmax=30.0,
        contact_cutoff=8.0,
        scaling_min_sep=5,
    ):
        """
        Executes all analysis methods and aggregates results into a single dictionary.

        Returns:
            dict: Comprehensive analysis results.
        """
        print("Starting comprehensive protein analysis...")

        results = {}

        # 1. Basic Dimensions
        print("-> Computing dimensions (Rg, End-to-End)...")

        results["ms_end_to_end"] = self.compute_mean_squared_end_to_end_distance()
        results["ms_radius_of_gyration"] = (
            self.compute_mean_squared_radius_of_gyration()
        )

        # 2. Global Topology & Scaling
        print("-> Computing gyration tensor and scaling exponent...")
        results["pooled_gyration_tensor"] = self.pooled_gyration_properties()
        results["scaling_exponent"] = self.compute_scaling_exponent(
            min_sep=scaling_min_sep
        )

        # 3. Secondary Structure
        print("-> Computing DSSP propensities...")
        results["secondary_structure"] = self.compute_secondary_structure_propensities()

        # 4. Backbone Grammar (Entropy/Dihedrals)
        print("-> Computing dihedral distributions and entropy...")
        results["pooled_dihedrals_entropy"] = self.pooled_dihedral_entropy()

        # 5. Network Dynamics & Correlations
        print("-> Computing DCCM and distance fluctuations...")
        results["dccm"] = self.compute_dccm()
        results["distance_fluctuations"] = self.compute_distance_fluctuations()

        # 6. Geometric Features
        print("-> Computing contact map...")
        results["contact_map"] = self.compute_contact_map(cutoff=contact_cutoff)

        # 7. Solvent-Solute Interactions
        print("-> Computing SASA and hydration density (this may take a while)...")
        results["residue_sasa"] = self.compute_residue_sasa(
            n_sphere_points=sasa_n_sphere, stride=sasa_stride
        )
        results["hydration_density"] = self.hydration_density_on_sequence(
            bins=hydration_bins, Rmax=hydration_rmax, cutoff=contact_cutoff
        )

        print("Analysis complete.")
        return results
