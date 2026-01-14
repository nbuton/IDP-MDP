import MDAnalysis
from tqdm import tqdm
import nmrgnn
import nmrdata
import numpy as np
import tensorflow as tf
from MDAnalysis.lib.nsgrid import FastNS
from scipy.spatial import cKDTree
import time

# Add this right after loading the Universe
from MDAnalysis.topology.guessers import guess_types

import numpy as np
import tensorflow as tf


def prepare_disjoint_batch(atoms_list, nlist_list, edges_list, inv_deg_list):
    """
    Combines multiple frames into a single disjoint graph for GNN batching.
    """
    total_atoms = 0
    all_atoms = []
    all_nlists = []
    all_edges = []
    all_inv_degs = []

    for i in range(len(atoms_list)):
        # 1. Collect atoms (usually static, but we repeat for the disjoint graph)
        all_atoms.append(atoms_list[i])

        # 2. Shift neighbor list indices by the current atom offset
        shifted_nlist = nlist_list[i] + total_atoms
        all_nlists.append(shifted_nlist)

        # 3. Edges and Inv_deg stay as they are (local to the interactions)
        all_edges.append(edges_list[i])
        all_inv_degs.append(inv_deg_list[i])

        # Update offset
        total_atoms += atoms_list[i].shape[0]

    # Concatenate everything into single large tensors
    return (
        tf.concat(all_atoms, axis=0),
        tf.concat(all_nlists, axis=0),
        tf.concat(all_edges, axis=0),
        tf.concat(all_inv_degs, axis=0),
    )


def parse_universe_fast(u, neighbor_number, embeddings, cutoff=None, pbc=None):
    # 1. Pre-calculate Atom Identities (Do this once outside the loop if possible!)
    # Cache the mapping of element -> integer ID
    if not hasattr(parse_universe_fast, "elem_map"):
        parse_universe_fast.elem_map = {k: v for k, v in embeddings["atom"].items()}

    N = u.atoms.n_atoms
    positions = u.atoms.positions

    # Fast Element Lookup
    try:
        elements = u.atoms.elements
    except:
        # Vectorized string cleaning: remove digits from all names at once
        elements = np.array(
            ["".join([c for c in n if not c.isdigit()]) for n in u.atoms.names]
        )

    # Map elements to IDs using a vectorized approach or a simple list comp (faster than atom-by-atom loop)
    atom_ids = np.array([parse_universe_fast.elem_map.get(e, 1) for e in elements])

    # 2. Grid Search (MDAnalysis C-optimized)
    if pbc is None:
        pbc = u.dimensions is not None
    dimensions = u.dimensions

    if cutoff is None:
        # Simplified cutoff logic to avoid bbox overhead every frame
        cutoff = 10.0  # Default to 10A if not specified, or pre-calculate once

    gridsearch = FastNS(cutoff, positions, dimensions, pbc=pbc)
    results = gridsearch.self_search()

    # 3. Vectorized Neighbor List Padding/Sorting
    # MDAnalysis returns pairs (i, j) and distances.
    # We use these to build the fixed-size arrays without a Python loop.
    pairs = results.get_pairs()
    dist = results.get_pair_distances()

    # Initialize output arrays
    nlist = np.zeros((N, neighbor_number), dtype=np.int32)
    edges = np.zeros((N, neighbor_number), dtype=np.float32)

    # Efficiently fill arrays (This is the trickiest part to vectorize perfectly)
    # Using a structured sort or grouping is faster than 'for i in range(N)'
    for i in range(N):
        idx = pairs[:, 0] == i
        node_dists = dist[idx]
        node_neighs = pairs[idx, 1]

        # Sort and Truncate
        order = np.argsort(node_dists)[:neighbor_number]
        n_found = len(order)

        nlist[i, :n_found] = node_neighs[order]
        edges[i, :n_found] = node_dists[order]

    edges /= 10.0  # Angstrom to nm

    return tf.one_hot(atom_ids, len(embeddings["atom"])), edges, nlist


def universe2graph(u, embeddings, neighbor_number=16):
    """Convert universe into tuple of graph objects. Universe is presumed to be in Angstrom. Universe should have explicit hydrogens

    Returns tuple with: atoms (one-hot element identity), nlist (neighbor list indices), edges (neighbor list distances), and inv_degree (inverse of degree for each atom).
    """
    atoms, edges, nlist = parse_universe_fast(u, neighbor_number, embeddings)
    mask = np.ones_like(atoms)
    inv_degree = tf.squeeze(
        tf.math.divide_no_nan(
            1.0, tf.reduce_sum(tf.cast(nlist > 0, tf.float32), axis=1)
        )
    )
    return atoms, nlist, edges, inv_degree


def get_clean_atom_ids(atom_names, embedding_map):
    """Maps IUPAC protein atom names (CA, HN, HT1, SD) to element IDs."""
    ids = []
    for name in atom_names:
        # 1. Clean numbers and uppercase
        s = "".join([i for i in name if not i.isdigit()]).upper()
        # 2. Logic for elements
        if s.startswith("SD") or s.startswith("SG"):
            elem = "S"
        elif s.startswith("FE"):
            elem = "Fe"
        else:
            elem = s[0]  # Maps CA->C, HN->H, N->N, OT->O

        # 3. Lookup ID (fallback to 1 for unknown)
        ids.append(embedding_map.get(elem, 1))
    return np.array(ids, dtype=np.int32)


@tf.function(reduce_retracing=True)
def fast_predict(model, atoms_one_hot, nlist_list, edge_list, inv_deg_list):
    n_total = len(nlist_list)
    results = tf.TensorArray(tf.float32, size=n_total)

    # Log the start
    tf.print("▶ Starting Graph Execution for", n_total, "frames...")

    for i in tf.range(n_total):
        p = model(
            (atoms_one_hot, nlist_list[i], edge_list[i], inv_deg_list[i]),
            training=False,
        )
        results = results.write(i, p)

        # Log every 100 frames (prevents terminal flooding)
        if i % 100 == 0:
            tf.print("  [Progress]: frame", i, "/", n_total)

    tf.print("✔ Prediction Complete.")
    return results.stack()


def compute_chemical_shift(u, n_samples):
    # tf.debugging.set_log_device_placement(True)
    # 1. SETUP & LOAD
    model = nmrgnn.load_model()
    embeddings = nmrdata.load_embeddings()
    print(f"Number of residues: {len(u.residues)}")
    elem_map = embeddings["atom"]

    # 2. PRE-PROCESS TOPOLOGY (Static)
    print("Mapping elements and indexing atoms...")
    atom_ids = get_clean_atom_ids(u.atoms.names, elem_map)
    # nmrgnn typically expects a depth of 10 for the one-hot vector
    atoms_one_hot = tf.one_hot(atom_ids, depth=10).numpy()

    # Identify target atoms for accumulation
    target_names = ["C", "CA", "CB", "N", "H"]
    # Handle the H/HN/HT naming variation you have
    target_selections = {
        "C": u.select_atoms("name C"),
        "CA": u.select_atoms("name CA"),
        "CB": u.select_atoms("name CB"),
        "N": u.select_atoms("name N"),
        "H": u.select_atoms("name H or name HN or name HT*"),
    }

    # Map (Residue_Index, Atom_Type_Index) -> Universe_Atom_Index
    res_map = {res.resid: i for i, res in enumerate(u.residues)}
    mapping_list = []
    for a_type_idx, t_name in enumerate(target_names):
        for atom in target_selections[t_name]:
            if atom.resid in res_map:
                mapping_list.append((res_map[atom.resid], a_type_idx, atom.index))

    # 3. TRAJECTORY PROCESSING (The "Super-Batch")
    frames = u.trajectory[
        sorted(np.random.choice(len(u.trajectory), n_samples, replace=False))
    ]
    n_frames = len(frames)
    n_atoms = u.atoms.n_atoms

    # Pre-allocate batch arrays
    all_nlist = np.zeros((n_frames, n_atoms, 16), dtype=np.int32)
    all_edges = np.zeros((n_frames, n_atoms, 16), dtype=np.float32)
    all_inv_deg = np.zeros((n_frames, n_atoms), dtype=np.float32)

    print(f"Processing {n_frames} frames geometry...")
    for i, ts in enumerate(tqdm(frames)):
        # Use KDTree for instant neighbor search (16 nearest neighbors)
        tree = cKDTree(u.atoms.positions)
        # k=17 because index 0 is the atom itself
        dist, indices = tree.query(u.atoms.positions, k=17)

        # Scale to nanometers and store
        all_nlist[i] = indices[:, 1:]
        all_edges[i] = dist[:, 1:] / 10.0

        # Calculate inverse degree
        deg = np.sum(all_nlist[i] >= 0, axis=1)
        all_inv_deg[i] = np.where(deg > 0, 1.0 / deg, 0)

    # 4. PREDICTION LOOP
    print("Start the prediction")
    start_time = time.time()
    all_preds = []
    with tf.device("/GPU:0"):
        atoms_tensor = tf.convert_to_tensor(atoms_one_hot, dtype=tf.float32)
        nlist_tensor = tf.convert_to_tensor(all_nlist, dtype=tf.int32)
        edges_tensor = tf.convert_to_tensor(all_edges, dtype=tf.float32)
        inv_deg_tensor = tf.convert_to_tensor(all_inv_deg, dtype=tf.float32)

    with tf.device("/GPU:0"):
        # This "warms up" the model on the GPU
        # Most Keras models move their weights to the GPU the first time they are called
        # within a GPU context.
        all_preds = fast_predict(
            model, atoms_tensor, nlist_tensor, edges_tensor, inv_deg_tensor
        )

    print(f"End the prediction, it took {time.time()-start_time}")
    all_peaks = np.squeeze(np.array(all_preds))  # Shape: (Frames, Atoms)

    # 5. ACCUMULATION
    print("Averaging shifts...")
    sum_shifts = np.zeros((len(u.residues), 5))
    counts = np.zeros((len(u.residues), 5))

    for r_idx, a_type_idx, u_idx in mapping_list:
        vals = all_peaks[:, u_idx]
        valid = vals > 0.1  # Ignore zeros/masking
        if np.any(valid):
            sum_shifts[r_idx, a_type_idx] += np.sum(vals[valid])
            counts[r_idx, a_type_idx] += np.sum(valid)

    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.true_divide(sum_shifts, counts)
        averaged[~np.isfinite(averaged)] = np.nan

    return {f"cs_{n.lower()}": averaged[:, i] for i, n in enumerate(target_names)}


stride_subsample = 5
idp_dir = (
    "data/IDRome/IDRome_v4/A4/D1/26/1_50/"  # "data/IDRome/IDRome_v4/Q5/SV/97/1_790"
)
md_analysis_u = MDAnalysis.Universe(idp_dir + "/top_AA.pdb", idp_dir + "/traj_AA.xtc")

start_time = time.time()
res_stride_1010 = compute_chemical_shift(md_analysis_u, n_samples=400)
print("Elapsed time:", time.time() - start_time)
start_time = time.time()
res_stride_400 = compute_chemical_shift(md_analysis_u, n_samples=400)
print("Elapsed time:", time.time() - start_time)

print(f"{'Atom':<6} | {'RMSD (1010 vs 400)':<15} | {'Max Diff':<10}")
print("-" * 40)

for key in res_stride_1010.keys():
    # Filter out NaNs to calculate difference
    mask = ~np.isnan(res_stride_1010[key]) & ~np.isnan(res_stride_400[key])
    diff = res_stride_1010[key][mask] - res_stride_400[key][mask]

    rmsd = np.sqrt(np.mean(diff**2))
    max_d = np.max(np.abs(diff))

    print(f"{key:<6} | {rmsd:<15.4f} | {max_d:<10.4f}")
