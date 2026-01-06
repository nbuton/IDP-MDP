from joblib import Parallel, delayed
import os
from idpmdp.utils import get_pdb_directories
from idpmdp.protein_analyzer import ProteinAnalyzer
from pathlib import Path


def compute_all_for_one_protein(directory_path):
    output_file = directory_path / "properties.h5"
    # Skip if the file already exists
    if output_file.exists():
        return f"Skipping {directory_path.name}: properties.h5 already exists."

    pdb_path = directory_path / "top_AA.pdb"
    xtc_path = directory_path / "traj_AA.xtc"

    analyzer = ProteinAnalyzer(pdb_path, xtc_path)
    results = analyzer.compute_all(
        sasa_n_sphere=960,
        sasa_stride=1,
        contact_cutoff=8.0,
        scaling_min_sep=5,
    )
    analyzer.save_all(results, output_folder=directory_path)
    return f"Successfully processed {directory_path.name}"


if __name__ == "__main__":
    all_idp_dir = get_pdb_directories("data/IDRome/IDRome_v4/")
    n_jobs = 30
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(compute_all_for_one_protein)(directory_path)
        for directory_path in all_idp_dir
    )
