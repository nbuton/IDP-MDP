import os
from cg2all import convert_cg2all
import cg2all.lib.libmodel
from cg2all.lib.libconfig import MODEL_HOME
import time
import os
import subprocess
import mdtraj as md
import numpy as np
from openmm.app import PDBFile, ForceField, Simulation, PME, HBonds
from openmm import LangevinMiddleIntegrator, OpenMMException, Platform
from pathlib import Path
from tqdm import tqdm

# Download the model weights for cg2all
for model_type in [
    "CalphaBasedModel",
    "ResidueBasedModel",
    "SidechainModel",
    "CalphaCMModel",
    "CalphaSCModel",
    "BackboneModel",
    "MainchainModel",
    "Martini",
    "Martini3",
    "PRIMO",
]:
    ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
    if not ckpt_fn.exists():
        cg2all.lib.libmodel.download_ckpt_file(model_type, ckpt_fn)


def create_backmap_file_idrome(
    idp_folder, is_idp=True, device="cuda", batch_cg2all=16, nb_proc_cg2all=4
):
    """
    Corrected backmapping following the CALVADOS/IDRome protocol.
    Converts .xtc to .dcd temporarily to ensure compatibility with cg2all CLI.
    """
    # 1. Define Standard IDRome Paths
    # IDRome typically uses 'top.pdb' for the coarse-grained topology
    cg_pdb = os.path.join(idp_folder, "top.pdb")
    cg_xtc = os.path.join(idp_folder, "traj.xtc")

    # Internal temporary file (cg2all works best with .dcd trajectories)
    cg_dcd = os.path.join(idp_folder, "traj_temp.dcd")

    # Outputs
    out_pdb = os.path.join(idp_folder, "top_AA.pdb")
    out_xtc = os.path.join(idp_folder, "traj_AA.dcd")

    if not os.path.exists(cg_pdb) or not os.path.exists(cg_xtc):
        raise FileNotFoundError(f"Missing {cg_pdb} or {cg_xtc}")

    # 2. Pre-process: Convert XTC to DCD (Internal requirement for the tool)
    # The notebook logic assumes DCD format
    traj = md.load(cg_xtc, top=cg_pdb)
    traj.save_dcd(cg_dcd)

    # 3. Model Selection
    # CalphaBasedModel for IDPs, ResidueBasedModel for MDPs
    cg_model = "CalphaBasedModel" if is_idp else "ResidueBasedModel"
    print(f"--- Starting Reconstruction using {cg_model} ---")

    # 4. Run the conversion via CLI (The notebook's preferred method)
    cmd = [
        "convert_cg2all",
        "-p",
        cg_pdb,
        "-d",
        cg_dcd,
        "-o",
        out_xtc,
        "-opdb",
        out_pdb,
        "--cg",
        cg_model,
        "--device",
        device,
        "--batch",
        str(batch_cg2all),
        "--proc",
        str(nb_proc_cg2all),
    ]
    print("I will run the command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 5. Energy Minimization
    # This step is vital to fix clashing atoms created during backmapping
    print("--- Minimizing All-Atom Structure ---")
    start_time_loading = time.time()
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    print("Time loading forcefield:", time.time() - start_time_loading, "s")
    pdb_em = PDBFile(out_pdb)

    system = forcefield.createSystem(
        pdb_em.topology, nonbondedMethod=PME, nonbondedCutoff=1.0, constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(300, 1, 0.004)

    print("Force field loading time:", time.time() - start_time_loading, "s")
    if device == "cuda":
        sim = Simulation(
            pdb_em.topology, system, integrator, Platform.getPlatformByName("OpenCL")
        )
    else:
        sim = Simulation(
            pdb_em.topology, system, integrator, Platform.getPlatformByName("CPU")
        )

    sim.context.setPositions(pdb_em.positions)
    sim.minimizeEnergy()

    # Save the finalized minimized PDB
    state = sim.context.getState(getPositions=True)
    final_traj = md.load_pdb(out_pdb)
    md.Trajectory(state.getPositions(asNumpy=True)._value, final_traj.top).save_pdb(
        out_pdb
    )

    # Cleanup temp files
    if os.path.exists(cg_dcd):
        os.remove(cg_dcd)

    print(f"Conversion complete. Final All-Atom PDB: {out_pdb}")


def get_pdb_directories(root_path):
    """
    Returns a set of unique Path objects representing directories
    that contain at least one .pdb file.
    """
    root = Path(root_path)
    pdb_dirs = {p.parent for p in root.rglob("*.pdb") if p.is_file()}
    return pdb_dirs


# Usage
if __name__ == "__main__":
    device = "cuda"
    print("Device:", device)
    root_path = "data/IDRome/IDRome_v4/A0"
    all_folder = get_pdb_directories(root_path)
    for folder in tqdm(all_folder):
        start_time = time.time()
        create_backmap_file_idrome(
            folder,
            is_idp=True,
            device=device,
            batch_cg2all=16,
            nb_proc_cg2all=4,
        )
        print("Elapsed time:", time.time() - start_time, "s")
