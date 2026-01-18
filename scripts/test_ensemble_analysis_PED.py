from idpmdp.analysis.orchestrator import ProteinAnalyzer
import logging
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test PED loading
    print("Analysing ensemble from PED")
    list_pdb_path = [
        "data/PED/PED00001/e001_ensemble-pdb.pdb",
        "data/PED/PED00001/e002_ensemble-pdb.pdb",
        "data/PED/PED00001/e003_ensemble-pdb.pdb",
    ]
    for pdb_path in list_pdb_path:
        analyzer = ProteinAnalyzer(Path(pdb_path), xtc_path=None, from_PED=True)
        results = analyzer.compute_all(
            sasa_n_sphere=250,
            contact_cutoff=8.0,
            scaling_min_sep=5,
        )
        print(results.keys())
