from idpmdp.protein_analyzer import ProteinAnalyzer
import numpy as np
import time


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
                print(f"{spacing}{key}: Array {value.shape}")
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
    start_time = time.time()
    pdb_path = "data/ATLAS/1k5n_A_analysis/1k5n_A.pdb"
    xtc_path = "data/ATLAS/1k5n_A_analysis/1k5n_A_R2.xtc"

    analyzer = ProteinAnalyzer(pdb_path, xtc_path)
    results = analyzer.compute_all(
        sasa_n_sphere=250,
        sasa_stride=10,
        hydration_bins=50,
        hydration_rmax=30.0,
        contact_cutoff=8.0,
        scaling_min_sep=5,
    )
    print_results(results)
    print(f"\nTotal analysis time: {time.time() - start_time:.2f} seconds")
