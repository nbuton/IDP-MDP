import os
from cg2all import convert_cg2all
import cg2all.lib.libmodel
from cg2all.lib.libconfig import MODEL_HOME

import os
import subprocess
import mdtraj as md
import numpy as np
from openmm.app import PDBFile, ForceField, Simulation, PME, HBonds
from openmm import LangevinMiddleIntegrator, OpenMMException, Platform


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


def create_backmap_file_idrome(idp_folder, is_idp=True, device="cpu"):
    """
    Corrected backmapping following the CALVADOS/IDRome protocol.
    Converts .xtc to .dcd temporarily to ensure compatibility with cg2all CLI.
    """
    # 1. Define Standard IDRome Paths
    # IDRome typically uses 'top_CA.pdb' for the coarse-grained topology
    cg_pdb = os.path.join(idp_folder, "top_CA.pdb")
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
    ]
    subprocess.run(cmd, check=True)

    # 5. Energy Minimization
    # This step is vital to fix clashing atoms created during backmapping
    print("--- Minimizing All-Atom Structure ---")
    pdb_em = PDBFile(out_pdb)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb_em.topology, nonbondedMethod=PME, nonbondedCutoff=1.0, constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(300, 1, 0.004)

    try:
        sim = Simulation(
            pdb_em.topology, system, integrator, Platform.getPlatformByName("CUDA")
        )
    except:
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


# Usage
if __name__ == "__main__":
    create_backmap_file_idrome(
        "data/IDRome/IDRome_v4/A0/A0/24/RBG1/145_181", is_idp=True, device="cpu"
    )
