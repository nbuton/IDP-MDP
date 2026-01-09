import mdtraj as md
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_sequence_from_IDRome_folder(one_IDRome_IDP):
    pdb_file = one_IDRome_IDP / "top_AA.pdb"
    traj = md.load(pdb_file)
    topology = traj.topology
    fasta_sequences = topology.to_fasta()
    assert len(fasta_sequences) == 1
    return fasta_sequences[0]


def process_single_pdb(pdb_file):
    """Worker function to process one PDB file."""
    folder = pdb_file.parent
    parts = folder.parts

    # Reconstruct Full UniProt ID: e.g., A2 + A2 + 88 -> A2A288
    uniprot_id = "".join(parts[-4:-1])
    coords = parts[-1]  # e.g., '44_84'
    uniprot_id_pos = f"{uniprot_id}_{coords}"

    # Get Sequence
    sequence = get_sequence_from_IDRome_folder(folder)

    return {"uniprot_id_pos": uniprot_id_pos, "sequence": sequence}


def create_all_sequences_csv(root_dir, output_csv, max_workers=None):
    root_path = Path(root_dir)
    # Find all files first
    pdb_files = [f for f in tqdm(root_path.rglob("top.pdb"), desc="Files found")]

    results = []

    # Use ProcessPoolExecutor for CPU-heavy/IO tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the worker function to the file list
        futures = [executor.submit(process_single_pdb, f) for f in pdb_files]

        # Wrap with tqdm to see progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing sequences"
        ):
            res = future.result()
            if res:
                results.append(res)

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} sequences to {output_csv}")


def unify_selected_idrome_to_h5(root_dir, path_output_h5, selected_entries):
    """
    selected_entries: list of strings ['A2A288_44_84', 'P12345_10_100', ...]
    """
    root_path = Path(root_dir)

    with h5py.File(path_output_h5, "w") as master_f:
        print(f"Processing {len(selected_entries)} selected entries...")

        for entry_key in tqdm(selected_entries):
            # Parse the string: ID_START_STOP
            # We split from the right twice to get start and stop
            try:
                uniprot_id, start, stop = entry_key.rsplit("_", 2)
                coords = f"{start}_{stop}"
            except ValueError:
                print(f"Skipping invalid format: {entry_key}")
                continue

            # IDRome structure logic: A2A288 -> root/A2/A2/88/44_84/
            p1, p2, p3 = uniprot_id[0:2], uniprot_id[2:4], uniprot_id[4:]
            folder = root_path / p1 / p2 / p3 / coords
            h5_path = folder / "properties.h5"

            if not h5_path.exists():
                print(f"Warning: File not found for {entry_key} at {h5_path}")
                continue

            # Get Sequence (using your existing helper function)
            sequence = get_sequence_from_IDRome_folder(folder)

            # Create Group in Master H5
            group = master_f.create_group(entry_key)
            group.attrs["sequence"] = sequence

            # Copy data from individual H5 to Master H5
            with h5py.File(h5_path, "r") as source_f:
                for key, obj in source_f.items():
                    if isinstance(obj, h5py.Dataset):
                        data = obj[()]
                        if np.isscalar(data) or (
                            hasattr(data, "shape") and data.shape == ()
                        ):
                            group.create_dataset(key, data=data)
                        else:
                            group.create_dataset(
                                key, data=data, compression="gzip", compression_opts=4
                            )

    print(f"Success! Unified data saved to {path_output_h5}")


def get_asymmetric_splits_from_csv(
    csv_path,
    min_id,
    cov_ratio,
    cov_mode,
    cluster_mode,
    train_ratio,
    val_ratio,
    test_ratio,
):
    """
    Reads a CSV with 'uniprot_id_pos' and 'sequence', clusters at 40% identity,
    and returns a dict with all sequences for train and representatives for val/test.
    """
    # Load data
    df_input = pd.read_csv(csv_path)

    # Calculate split proportions
    sum_val_test = val_ratio + test_ratio
    relative_test_size = test_ratio / sum_val_test

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Paths for temporary files
        input_fasta = os.path.join(tmp_dir, "input.fasta")
        output_prefix = os.path.join(tmp_dir, "mmseqs_out")
        tmp_working = os.path.join(tmp_dir, "mmseqs_tmp")

        # 1. Write CSV to FASTA (MMseqs2 requirement)
        with open(input_fasta, "w") as f:
            for _, row in df_input.iterrows():
                f.write(f">{row['uniprot_id_pos']}\n{row['sequence']}\n")

        # 2. Run MMseqs2 easy-cluster
        cmd = [
            "mmseqs",
            "easy-cluster",
            input_fasta,
            output_prefix,
            tmp_working,
            "--min-seq-id",
            str(min_id),
            "-c",
            str(cov_ratio),
            "--cov-mode",
            str(cov_mode),
            "--cluster-mode",
            str(cluster_mode),
        ]
        subprocess.run(cmd, check=True)

        # 3. Load clustering results
        # file format: representative_ID \t member_ID
        cluster_df = pd.read_csv(
            f"{output_prefix}_cluster.tsv", sep="\t", names=["rep", "member"]
        )
        unique_clusters = cluster_df["rep"].unique()

        # 4. Split based on Cluster Representatives (prevent leakage)
        train_reps, temp_reps = train_test_split(
            unique_clusters, train_size=train_ratio, random_state=42
        )
        val_reps, test_reps = train_test_split(
            temp_reps, test_size=relative_test_size, random_state=42
        )

        # 5. Build the final dictionary
        # We return the 'uniprot_id_pos' values
        splits = {
            # TRAIN: Every sequence belonging to the training clusters
            "train": cluster_df[cluster_df["rep"].isin(train_reps)]["member"].tolist(),
            # VALIDATION: Only the representative of each validation cluster
            "validation": val_reps.tolist(),
            # TEST: Only the representative of each test cluster
            "test": test_reps.tolist(),
        }

        return splits
