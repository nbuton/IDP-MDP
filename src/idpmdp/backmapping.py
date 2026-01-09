import os
import subprocess
import mdtraj as md
import time
from idpmdp.utils import get_pdb_directories
import logging


class IDRomeBackmapper:
    """
    Backmaps IDP/MDP trajectories to all-atom representation.
    Ensures input files are never overwritten by using new filenames for outputs.
    """

    # Standard residue mapping to ensure cg2all recognizes amino acids
    RESIDUE_MAPPING = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "ALA",
        "Z": "GLY",  # Handling CALVADOS termini
    }

    def __init__(
        self,
        top_pdb,
        traj_xtc,
        is_idp=True,
        device="cpu",
        cg2all_batch_size=16,
        cg2all_nb_proc=4,
    ):
        """
        Args:
            top_pdb (str): Path to original coarse-grained PDB.
            traj_xtc (str): Path to original coarse-grained XTC.
            is_idp (bool): True for CalphaBasedModel, False for ResidueBasedModel.
            device (str): 'cuda' to use OpenCL platform, else uses CPU.
        """
        self.top_pdb = os.path.abspath(top_pdb)
        self.traj_xtc = os.path.abspath(traj_xtc)
        self.is_idp = is_idp
        self.cg2all_batch_size = cg2all_batch_size
        self.cg2all_nb_proc = cg2all_nb_proc
        self.device = device.lower()
        self.cg_model = "CalphaBasedModel" if is_idp else "ResidueBasedModel"

    def run(self, suffix="AA"):
        """
        Executes backmapping and saves results in the same directory as input
        with the specified suffix (default: _AA).
        """
        start_time = time.time()
        base_dir = os.path.dirname(self.top_pdb)

        # Output filenames (different from inputs)
        aa_pdb = os.path.join(base_dir, f"top_{suffix}.pdb")
        aa_dcd = os.path.join(base_dir, f"traj_{suffix}.dcd")
        aa_xtc = os.path.join(base_dir, f"traj_{suffix}.xtc")

        if os.path.exists(aa_pdb) and os.path.exists(aa_xtc):
            print(f"Skipping: All-atom files already exist in {base_dir}")
            return

        # Temporary files for intermediate steps
        fixed_pdb = os.path.join(base_dir, f"temp_fixed_cg_{suffix}.pdb")
        fixed_dcd = os.path.join(base_dir, f"temp_fixed_cg_{suffix}.dcd")

        # Safety check: Prevent accidental overwriting of input
        if aa_pdb == self.top_pdb or aa_xtc == self.traj_xtc:
            raise ValueError(
                "Output names would overwrite input! Use a different suffix."
            )

        try:
            print(f"--- 1. Preprocessing Topology ---")
            self._fix_topology(fixed_pdb, fixed_dcd)

            print(f"--- 2. Reconstructing All-Atom using {self.cg_model} ---")
            self._reconstruct(fixed_pdb, fixed_dcd, aa_pdb, aa_dcd)

            print(f"--- 3. Converting Trajectory to XTC ---")
            self._convert_to_xtc(aa_dcd, aa_pdb, aa_xtc)

            print(f"Done in {time.time()-start_time}s")

        finally:
            # Clean up all temporary files used during processing
            for f in [fixed_pdb, fixed_dcd, aa_dcd]:
                if os.path.exists(f):
                    os.remove(f)

    def _fix_topology(self, out_pdb, out_dcd):
        """Renames bead atoms to CA and residue names to 3-letter codes."""
        t = md.load(self.traj_xtc, top=self.top_pdb)
        fixed_top = md.Topology()
        chain = fixed_top.add_chain()
        for atom in t.top.atoms:
            old_res = atom.residue.name
            new_res = self.RESIDUE_MAPPING.get(old_res, old_res)
            res = fixed_top.add_residue(new_res, chain)
            fixed_top.add_atom("CA", element=md.element.carbon, residue=res)

        fixed_traj = md.Trajectory(
            t.xyz, fixed_top, t.time, t.unitcell_lengths, t.unitcell_angles
        )
        fixed_traj[0].save_pdb(out_pdb)
        fixed_traj.save_dcd(out_dcd)

    def _reconstruct(self, in_pdb, in_dcd, out_pdb, out_dcd):
        """Calls convert_cg2all command line tool."""
        cmd = [
            "convert_cg2all",
            "-p",
            in_pdb,
            "-d",
            in_dcd,
            "-o",
            out_dcd,
            "-opdb",
            out_pdb,
            "--cg",
            self.cg_model,
            "--batch",
            str(self.cg2all_batch_size),
            "--proc",
            str(self.cg2all_nb_proc),
            "--device",
            self.device,
        ]
        subprocess.run(cmd, check=True)

    def _convert_to_xtc(self, dcd_path, pdb_path, xtc_path):
        """Converts intermediate DCD to final XTC."""
        traj = md.load(dcd_path, top=pdb_path)
        traj.save_xtc(xtc_path)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    all_idp_dir = get_pdb_directories("data/IDRome/IDRome_v4/")
    num_directories = len(all_idp_dir)
    processed_count = 0

    for directory_path in all_idp_dir:
        backmapper = IDRomeBackmapper(
            directory_path / "top.pdb",
            directory_path / "traj.xtc",
            is_idp=True,
            device="cuda",
            cg2all_batch_size=8,
            cg2all_nb_proc=4,
        )
        backmapper.run()
        processed_count += 1
        logging.info(f"Processed {processed_count}/{num_directories} directories")
