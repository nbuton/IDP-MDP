from joblib import Parallel, delayed
from idpmdp.utils import get_pdb_directories
from idpmdp.analysis.orchestrator import ProteinAnalyzer
from idpmdp.analysis.io_utils import save_all_properties


def compute_all_for_one_protein(directory_path):
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


if __name__ == "__main__":
    all_idp_dir = get_pdb_directories("data/IDRome/IDRome_v4/")
    n_jobs = 30
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(compute_all_for_one_protein)(directory_path)
        for directory_path in all_idp_dir
    )
