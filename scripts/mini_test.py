from idpmdp.analysis.orchestrator import ProteinAnalyzer
import numpy as np
import time
import logging
from pathlib import Path
from idpmdp.analysis.io_utils import save_all_properties


def count_total_floats(data):
    """
    Recursively counts the total number of elements in np.arrays
    within a nested structure of dicts and lists.
    """
    total_count = 0

    # Case 1: The item is a dictionary, iterate through its values
    if isinstance(data, dict):
        for value in data.values():
            total_count += count_total_floats(value)

    # Case 2: The item is a list or tuple, iterate through elements
    elif isinstance(data, (list, tuple)):
        for item in data:
            total_count += count_total_floats(item)

    # Case 3: The item is a NumPy array
    elif isinstance(data, np.ndarray):
        # We only count if the array contains float types
        if np.issubdtype(data.dtype, np.floating):
            total_count += data.size

    return total_count


def print_results(data, indent=0):
    """
    Recursively prints keys and shapes/values with clean indentation.
    """
    spacing = "  " * indent

    # If it's a dictionary, iterate through its items
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle specific types for the label
            if isinstance(value, np.ndarray):
                print(f"{spacing}{key}: Array {value.shape} : Dtype: {value.dtype}")
            elif isinstance(value, (list, tuple)):
                # Peek at the first element to check if it's a list of arrays
                if len(value) > 0 and isinstance(value[0], np.ndarray):
                    print(
                        f"{spacing}{key}: List of {len(value)} arrays, shape {value[0].shape}"
                    )
                else:
                    print(f"{spacing}{key}: List/Tuple (len {len(value)})")
            elif isinstance(value, dict):
                print(f"{spacing}{key}:")
                print_results(value, indent + 1)
            else:
                # For scalars (int, float, str)
                print(f"{spacing}{key}: {value}")
    else:
        print(f"{spacing}{data}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test IDRome
    print("Analysing ensemble from IDRome")
    start_time = time.time()
    IDP_folder = Path(
        "./data/IDRome/IDRome_v4/Q5/SY/C1/270_327"
    )  # 58 residues: ./data/IDRome/IDRome_v4/Q5/SY/C1/270_327 / 1485 residues: "./data/IDRome/IDRome_v4/Q5/JP/B2/193_1677"
    pdb_path = Path(IDP_folder / "top_AA.pdb")
    xtc_path = Path(IDP_folder / "traj_AA.xtc")
    analyzer = ProteinAnalyzer(pdb_path, xtc_path)
    results = analyzer.compute_all(
        sasa_n_sphere=100,  # 960
        sasa_stride=10,  # 1
        contact_cutoff=8.0,
        scaling_min_sep=5,
    )
    save_all_properties(results, output_folder=IDP_folder)
    print_results(results)
    print("dccm:", results["dccm"][:5, :5])
    print("distance_fluctuations:", results["distance_fluctuations"][:5, :5])
    print("contact_map:", results["contact_map"][:5, :5])
    print(f"\nTotal analysis time: {time.time() - start_time:.2f} seconds")
    exit(1)

    # Test PED loading
    print("Analysing ensemble from PED")
    pdb_path = [
        "data/PED/PED00001/e001_ensemble-pdb.pdb",
        "data/PED/PED00001/e002_ensemble-pdb.pdb",
        "data/PED/PED00001/e003_ensemble-pdb.pdb",
    ]
    analyzer = ProteinAnalyzer(pdb_path)
    results = analyzer.compute_all(
        sasa_n_sphere=250,
        sasa_stride=10,
        contact_cutoff=8.0,
        scaling_min_sep=5,
    )
    print_results(results)
    print("There are {count_total_floats(results)} elements in the results dictionary.")
