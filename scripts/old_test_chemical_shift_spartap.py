import mdtraj as md
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time

import mdtraj as md
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm


def compute_sparta_mdtraj(md_traj, stride=10):
    # 1. SETUP PATHS & ENVIRONMENT
    sparta_base = (
        "/srv/storage/capsid@srv-data2.nancy.grid5000.fr/nbuton/SPARTA_plus/SPARTA+"
    )
    sparta_bin = os.path.join(sparta_base, "bin")

    # MDTraj looks for "SPARTA+" in the system PATH
    os.environ["PATH"] = sparta_bin + os.pathsep + os.environ["PATH"]
    # SPARTA+ needs this to find its internal databases
    os.environ["SPARTAPLUSDIR"] = sparta_base

    # 2. SETUP RAM DISK
    ram_disk = "/dev/shm/mdt_sparta"
    os.makedirs(ram_disk, exist_ok=True)
    original_cwd = os.getcwd()

    all_shifts = []

    try:
        # Move to RAM disk to avoid collisions and speed up I/O
        os.chdir(ram_disk)

        # Iterate through the trajectory
        for i in tqdm(range(0, len(md_traj), stride), desc="MDTraj SPARTA+"):
            chunk = md_traj[i]

            try:
                # MDTraj wrapper for SPARTA+
                # It writes 'trj0.pdb' and expects 'trj0_pred.tab'
                shifts = md.nmr.chemical_shifts_spartaplus(chunk)
                all_shifts.append(shifts[0])
            except Exception as e:
                # Cleanup if a specific frame fails
                continue
            finally:
                # Cleanup temp files created by MDTraj in this loop
                for f in ["trj0.pdb", "trj0_pred.tab"]:
                    if os.path.exists(f):
                        os.remove(f)

    finally:
        # ALWAYS move back to the original directory
        os.chdir(original_cwd)

    if not all_shifts:
        raise ValueError(
            "No shifts collected. Verify that /srv/.../bin/SPARTA+ is executable."
        )

    # 3. PROCESSING RESULTS
    df = pd.DataFrame(all_shifts)
    mean_shifts = df.mean().reset_index()
    mean_shifts.columns = ["res_tuple", "shift"]

    # Extract Residue Index and Atom Name
    mean_shifts["RESID"] = mean_shifts["res_tuple"].apply(lambda x: x[0])
    mean_shifts["ATOMNAME"] = mean_shifts["res_tuple"].apply(lambda x: x[1])

    # Pivot and adjust to 1-based indexing
    final_df = mean_shifts.pivot(index="RESID", columns="ATOMNAME", values="shift")
    final_df.index = final_df.index + 1

    # Ensure we only return the standard backbone/beta atoms
    cols = [c for c in ["C", "CA", "CB", "N", "H"] if c in final_df.columns]
    return final_df[cols]


# Usage
idp_dir = "data/IDRome/IDRome_v4/Q5/JP/B2/193_1677/"
md_traj = md.load(idp_dir + "top_AA.pdb", idp_dir + "traj_AA.xtc")
start_time = time.time()
df_sparta = compute_sparta_mdtraj(md_traj, stride=10)
print(df_sparta.head())
print(f"Elapsed time: {time.time()-start_time}s")
