from joblib import Parallel, delayed
import os
from idpmdp.utils import get_pdb_directories
from idpmdp.protein_analyzer import ProteinAnalyzer
from pathlib import Path

if __name__ == "__main__":
    all_idp_dir = get_pdb_directories("data/IDRome/IDRome_v4/")

    for directory_path in all_idp_dir:
        pdb_path = directory_path / "top_AA.pdb"
        xtc_path = directory_path / "traj_AA.xtc"
        analyzer = ProteinAnalyzer(pdb_path, xtc_path)
        results = analyzer.compute_all(
            sasa_n_sphere=960,
            sasa_stride=1,
            contact_cutoff=8.0,
            scaling_min_sep=5,
        )
        # TODO: Save results to file
