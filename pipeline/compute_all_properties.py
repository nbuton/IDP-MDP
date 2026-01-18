from joblib import Parallel, delayed
from idpmdp.utils import get_pdb_directories
from idpmdp.analysis.orchestrator import ProteinAnalyzer
from idpmdp.analysis.io_utils import save_all_properties
import os
from pathlib import Path
import logging


def compute_all_for_one_protein_from_IDRome(directory_path):
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    output_file = directory_path / "properties.h5"
    # Skip if the file already exists
    if output_file.exists():
        return f"Skipping {directory_path.name}: properties.h5 already exists."

    try:
        pdb_path = directory_path / "top_AA.pdb"
        xtc_path = directory_path / "traj_AA.xtc"

        analyzer = ProteinAnalyzer(pdb_path, xtc_path)
        results = analyzer.compute_all(
            sasa_n_sphere=960,
            contact_cutoff=8.0,
            scaling_min_sep=5,
        )
        save_all_properties(results, output_folder=directory_path)
        return f"SUCCESS: {directory_path.name}"

    except Exception as e:
        # It's good practice to log the error or return it for later inspection
        return f"FAILED: {directory_path.name} with error: {str(e)}"


def compute_all_for_one_protein_from_PED(directory_path):
    all_pdb_ensembles = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".pdb")
    ]

    for pdb_ensemble_set in all_pdb_ensembles:
        pdb_ensemble_set = Path(pdb_ensemble_set)
        output_folder = directory_path / pdb_ensemble_set.stem
        output_file = output_folder / "properties.h5"
        # Skip if the file already exists
        if output_file.exists():
            return f"Skipping {directory_path.name}: properties.h5 already exists."
        try:
            analyzer = ProteinAnalyzer(pdb_ensemble_set, xtc_path=None, from_PED=True)
            results = analyzer.compute_all(
                sasa_n_sphere=960,
                contact_cutoff=8.0,
                scaling_min_sep=5,
            )
        except:
            print("Not suitable ensemble")
            continue

        print(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        save_all_properties(results, output_folder=output_folder)

    return f"SUCCESS: {directory_path.name}"


if __name__ == "__main__":
    parallel = True
    n_jobs = 1

    # all_idp_dir = get_pdb_directories("data/IDRome/IDRome_v4/")
    # if parallel:
    #     results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
    #         delayed(compute_all_for_one_protein_from_IDRome)(directory_path)
    #         for directory_path in all_idp_dir
    #     )
    # else:
    #     for idp_dir in all_idp_dir:
    #         print(compute_all_for_one_protein_from_IDRome(idp_dir))

    all_idp_dir = get_pdb_directories("data/PED/")
    if parallel:
        results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
            delayed(compute_all_for_one_protein_from_PED)(directory_path)
            for directory_path in all_idp_dir
        )
    else:
        for idp_dir in all_idp_dir:
            print(compute_all_for_one_protein_from_PED(idp_dir))
