import MDAnalysis
from MDAnalysis.analysis import distances
import numpy as np
from scipy.spatial.distance import pdist
import mdtraj as md
from scipy.stats import linregress


def compute_end_to_end_distance(
    md_analysis_u: MDAnalysis.Universe, residues: MDAnalysis.ResidueGroup
):
    """
    Calculates the distance between the CA atoms of the
    first and last residues across the trajectory.
    """
    # Inside compute_end_to_end_distance
    all_ca = md_analysis_u.select_atoms("name CA")

    start_ca = all_ca[0:1]
    end_ca = all_ca[-1:]

    # Ensure single atom selections
    assert (
        len(start_ca) == 1
    ), f"Start residue CA atom not found. Available atoms names: {residues[0].atoms.names}"
    assert (
        len(end_ca) == 1
    ), f"End residue CA atom not found.Available atoms names: {residues[0].atoms.names}"

    all_distances = []
    for ts in md_analysis_u.trajectory:
        resA, resB, dist = distances.dist(start_ca, end_ca)
        all_distances.append(dist)

    return np.array(all_distances)


def compute_maximum_diameter(md_analysis_u):
    """
    Calculates the maximum distance between any two CA atoms
    (Dmax) for each frame in the trajectory.
    """
    # 1. Pre-select CA atoms to avoid overhead inside the loop
    ca_atoms = md_analysis_u.select_atoms("name CA")

    all_dmax = []

    # 2. Iterate through trajectory
    for ts in md_analysis_u.trajectory:
        # Get the coordinates for the current frame
        coords = ca_atoms.positions

        # pdist computes the distance between every pair of atoms (n*(n-1)/2 distances)
        # np.max gives us the largest of those distances
        dmax = np.max(pdist(coords))
        all_dmax.append(dmax)

    return np.array(all_dmax)


def compute_gyration_tensor_properties(md_traj):
    results = {
        "gyration_eigenvalues_l1": [],
        "gyration_eigenvalues_l2": [],
        "gyration_eigenvalues_l3": [],
        "gyration_l1_per_l2": [],
        "gyration_l1_per_l3": [],
        "gyration_l2_per_l3": [],
        "radius_of_gyration": [],
        "asphericity": [],
        "normalized_acylindricity": [],
        "rel_shape_anisotropy": [],
        "prolateness": [],
    }

    protein_traj = md_traj.atom_slice(md_traj.topology.select("protein"))

    eigvals = md.principal_moments(protein_traj)

    for one_set_of_eigvals in eigvals:
        l1, l2, l3 = one_set_of_eigvals
        results["gyration_eigenvalues_l1"].append(l1)
        results["gyration_eigenvalues_l2"].append(l2)
        results["gyration_eigenvalues_l3"].append(l3)
        results["gyration_l1_per_l2"].append(l1 / l2 if l2 > 0 else 0)
        results["gyration_l1_per_l3"].append(l1 / l3 if l3 > 0 else 0)
        results["gyration_l2_per_l3"].append(l2 / l3 if l3 > 0 else 0)

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
        denominator_p = 2 * (l1**2 + l2**2 + l3**2 - l1 * l2 - l1 * l3 - l2 * l3) ** 1.5
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


def compute_scaling_exponent(md_traj, min_sep=5):
    """Scaling exponent ν using self.md_traj - Cα selection inside."""

    # Select only Cα atoms
    ca_mask = md_traj.topology.select("name CA")
    traj_ca = md_traj.atom_slice(ca_mask)
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
