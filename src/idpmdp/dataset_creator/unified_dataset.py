import mdtraj as md
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np


def get_sequence_from_IDRome_folder(one_IDRome_IDP):
    pdb_file = one_IDRome_IDP / "top_AA.pdb"
    traj = md.load(pdb_file)
    topology = traj.topology
    fasta_sequences = topology.to_fasta()
    assert len(fasta_sequences) == 1
    return fasta_sequences[0]


def unify_idrome_to_h5(root_dir, path_output_h5):
    root_path = Path(root_dir)
    h5_files = list(root_path.rglob("properties.h5"))

    with h5py.File(path_output_h5, "w") as master_f:
        print(f"Processing {len(h5_files)} files...")

        for h5_path in tqdm(h5_files):
            folder = h5_path.parent
            parts = folder.parts

            # Reconstruct Full UniProt ID: e.g., A2 + A2 + 88 -> A2A288
            uniprot_id = "".join(parts[-4:-1])
            coords = parts[-1]  # e.g., '44_84'
            entry_key = f"{uniprot_id}_{coords}"

            # Get Sequence
            sequence = get_sequence_from_IDRome_folder(folder)

            # Create a Group for this specific protein entry
            group = master_f.create_group(entry_key)
            group.attrs["sequence"] = sequence

            with h5py.File(h5_path, "r") as source_f:
                # .items() returns (name, object) pairs
                for key, obj in source_f.items():
                    print(type(obj))
                    print(obj)
                    # Check if it's a dataset (and not a group)
                    if isinstance(obj, h5py.Dataset):
                        data = obj[()]  # Load data into RAM
                        if np.isscalar(data) or (
                            hasattr(data, "shape") and data.shape == ()
                        ):
                            group.create_dataset(key, data=data)
                        else:
                            # Apply compression only to vectors and matrices
                            group.create_dataset(
                                key, data=data, compression="gzip", compression_opts=4
                            )
                    else:
                        print("Not ok")
                        # raise ValueError("Not correct object type")

    print(f"Success! Unified data saved to {path_output_h5}")


if __name__ == "__main__":
    # Usage
    root_directory = "data/IDRome/IDRome_v4/"
    unify_idrome_to_h5(root_directory, path_output_h5="data/unified_dataset.h5")
