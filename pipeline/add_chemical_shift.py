from joblib import Parallel, delayed
from idpmdp.utils import get_pdb_directories
from idpmdp.analysis.residue_level_metrics import compute_chemical_shift
from idpmdp.analysis.io_utils import (
    save_all_properties,
    load_all_properties,
)
import MDAnalysis
import os


def update_chemical_shifts(directory_path):
    output_file = directory_path / "properties.h5"
    lock_file = directory_path / "processing.lock"

    if not output_file.exists():
        return f"SKIP: {directory_path.name} (No existing properties.h5 found)"

    # 1. Load existing results
    # Assuming load_all_properties returns a dict of dataframes/arrays
    results = load_all_properties(output_file)

    if lock_file.exists():
        return f"SKIP: {directory_path.name} (Currently being processed by another instance)"

    try:
        # 'x' mode fails if the file already exists (atomic-like operation)
        with open(lock_file, "w") as f:
            f.write("locked")
    except FileExistsError:
        return f"SKIP: {directory_path.name} (Race condition caught)"

    # 2. Check if chemical_shifts already exists to avoid redundant work
    if "chemical_shifts" in results:
        return f"SKIP: {directory_path.name} (Chemical shifts already present)"

    try:
        # 3. Initialize analyzer and compute ONLY the missing property
        pdb_path = directory_path / "top_AA.pdb"
        xtc_path = directory_path / "traj_AA.xtc"

        md_analysis_u = MDAnalysis.Universe(pdb_path, xtc_path)

        # Compute chemical shifts
        cs_dict_results = compute_chemical_shift(md_analysis_u, pdb_path, batch_size=96)

        # 4. Update the results dictionary and save
        results.update(cs_dict_results)
        save_all_properties(results, output_folder=directory_path)

        return f"SUCCESS: Updated {directory_path.name} with chemical shifts"
    finally:
        # 3. ALWAYS REMOVE LOCK (even if script crashes)
        if lock_file.exists():
            os.remove(lock_file)


if __name__ == "__main__":
    all_idp_dir = get_pdb_directories(
        "/srv/storage/capsid@srv-data2.nancy.grid5000.fr/nbuton/IDP-MDP/data/IDRome/IDRome_v4/"
    )
    n_jobs = 32  # 64

    print(f"Starting chemical shift update for {len(all_idp_dir)} directories...")
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(update_chemical_shifts)(directory_path)
        for directory_path in all_idp_dir
    )
