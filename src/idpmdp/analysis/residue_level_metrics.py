import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree


def has_overlapping_atoms(traj, threshold=0.001):
    """
    Checks all frames in an MDTraj object for overlapping atoms.
    threshold is in nanometers (default 0.001 nm = 0.01 Angstrom).
    """
    if traj.n_atoms < 2:
        return False

    # Iterate through frames
    for i in range(traj.n_frames):
        coords = traj.xyz[i]

        # Build tree for the current frame
        tree = cKDTree(coords)

        # query_pairs is very fast (written in C++)
        pairs = tree.query_pairs(r=threshold)

        if len(pairs) > 0:
            # We found a clash in at least one frame
            # Returning the frame index can help with debugging
            return True, i

    return False, None


def remove_problematic_frames(traj, frame_idx):
    """
    Removes specific frames from an md.Trajectory object.

    Parameters:
    - traj: The md.Trajectory object.
    - frame_idx: An integer or a list/set of integers representing the frames to remove.

    Returns:
    - A new md.Trajectory object without the problematic frames.
    """
    # Ensure frame_idx is a set for fast lookup and to handle single integers
    if isinstance(frame_idx, (int, np.integer)):
        to_remove = {frame_idx}
    else:
        to_remove = set(frame_idx)

    # Create a list of all frame indices EXCEPT those in the remove set
    all_indices = np.arange(traj.n_frames)
    keep_indices = [i for i in all_indices if i not in to_remove]

    if not keep_indices:
        print(f"Warning: Removing frames {to_remove} left the trajectory empty!")
        return None

    # Use MDTraj's built-in slicing to create the new trajectory
    # This is highly optimized and preserves topology
    clean_traj = traj[keep_indices]

    return clean_traj


def compute_secondary_structure_propensities(md_traj):
    """
    Computes the SS8 (DSSP) propensity of each secondary structure element
    per residue over the trajectory using MDTraj.
    """
    traj = md_traj.atom_slice(md_traj.top.select("protein"))

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
    assert np.all(np.diff(resids) == 1), f"Gap detected in residue sequence: {resids}"

    return {f"ss_propensity_{k}": v for k, v in propensities.items()}


def compute_dihedral_distribution(md_traj):
    """Uses MDTraj to compute Dihedrals and X-Entropy for S_conf."""

    phi_indices, phi_angles = md.compute_phi(md_traj)
    psi_indices, psi_angles = md.compute_psi(md_traj)

    n_frames = md_traj.n_frames
    n_residues = md_traj.topology.n_residues  # Total length L

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


def pooled_dihedral_entropy(md_traj, bins=60):
    """
    phi_array: shape (n_frames, n_residues)
    psi_array: shape (n_frames, n_residues)
    """
    results = compute_dihedral_distribution(md_traj)
    phi_array = results["phi"]
    psi_array = results["psi"]

    n_frames, n_res = phi_array.shape
    pooled_entropy = {"phi_dihedrals_entropy": [], "psi_dihedrals_entropy": []}

    for i in range(n_res):
        # Process PHI for residue i
        phi_counts, _ = np.histogram(phi_array[:, i], bins=bins, range=[-np.pi, np.pi])
        p_phi = phi_counts / n_frames
        p_phi = p_phi[p_phi > 0]  # Remove zeros to avoid log(0)
        s_phi = -np.sum(p_phi * np.log(p_phi))

        # Process PSI for residue i
        psi_counts, _ = np.histogram(psi_array[:, i], bins=bins, range=[-np.pi, np.pi])
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


def compute_local_chirality(md_traj):
    """
    Compute mean local chirality <χ_i> and std(χ_i) for all residues.
    """
    # 1. Correct way to get residue count
    n_res = md_traj.topology.n_residues

    # 2. Select C-alpha atom indices
    calphas = md_traj.topology.select("name CA")

    # Ensure we have enough residues to compute the scalar triple product
    if n_res < 4:
        raise ValueError(
            "Trajectory must have at least 4 residues to compute local chirality."
        )

    # 3. Extract all C-alpha coordinates for all frames: shape (n_frames, n_ca, 3)
    xyz = md_traj.xyz[:, calphas, :]

    # 4. Vectorized calculation of v1, v2, v3 for all residues and frames
    # v1: i-1 to i
    # v2: i to i+1
    # v3: i+1 to i+2
    v1 = xyz[:, 1:-2, :] - xyz[:, 0:-3, :]
    v2 = xyz[:, 2:-1, :] - xyz[:, 1:-2, :]
    v3 = xyz[:, 3:, :] - xyz[:, 2:-1, :]

    # 5. Compute scalar triple product: chi = v1 · (v2 × v3)
    # np.cross handles the (n_frames, n_valid_res, 3) arrays perfectly
    cross_v2_v3 = np.cross(v2, v3)
    # Sum over the last axis (the 3D coordinates) to perform the dot product
    chi_all = np.sum(v1 * cross_v2_v3, axis=2)

    # 6. Compute statistics over frames (axis 0)
    mean_chi = np.mean(chi_all, axis=0)
    std_chi = np.std(chi_all, axis=0)

    # Note: This returns values for residues index 1 to n_res-2
    return np.column_stack([mean_chi, std_chi])


def compute_residue_sasa(md_traj, n_sphere_points):
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

    # Select only the protein (standard practice to avoid membrane/solvent interference)
    has_overlap, frame_idx = has_overlapping_atoms(md_traj)
    if has_overlap:
        md_traj = remove_problematic_frames(md_traj, frame_idx)

    protein_indices = md_traj.topology.select("protein")
    t_prot = md_traj.atom_slice(protein_indices)

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
