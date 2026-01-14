import os
import tempfile
import numpy as np
import pandas as pd
import MDAnalysis as mda
import subprocess
from contextlib import contextmanager
import MDAnalysis
import time


if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    idp_dir = "data/IDRome/IDRome_v4/Q5/T7/B8/541_1292/"  # "data/IDRome/IDRome_v4/Q5/SY/C1/270_327/"  # "data/IDRome/IDRome_v4/A4/D1/26/1_50/" # "data/IDRome/IDRome_v4/Q5/JP/B2/193_1677"
    pdb_path = os.path.join(idp_dir, "top_AA.pdb")
    xtc_path = os.path.join(idp_dir, "traj_AA.xtc")

    if os.path.exists(pdb_path):
        start_time = time.time()
        md_analysis_u = MDAnalysis.Universe(pdb_path, xtc_path)
        results = compute_legolas_shifts(md_analysis_u, pdb_path, batch_size=32)
        print(f"Successfully computed shifts for {len(results)} atom types.")
        print(f"Elapsed time: {time.time()-start_time:.2f}s")
    else:
        print(f"File not found: {pdb_path}")
