import h5py
from pathlib import Path
import numpy as np


def save_all_properties(results, output_folder: Path):
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
