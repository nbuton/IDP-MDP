from idpmdp.analysis.orchestrator import ProteinAnalyzer
from idpmdp.backmapping import IDRomeBackmapper
import mdtraj as md
import numpy as np


def compute_basic_prop(IDP_folder, suffix):
    pdb_path = IDP_folder + "top" + suffix + ".pdb"
    xtc_path = IDP_folder + "traj" + suffix + ".xtc"
    analyzer = ProteinAnalyzer(pdb_path, xtc_path)
    end_to_end_dist = analyzer.compute_mean_squared_end_to_end_distance()
    print("end_to_end_dist:", end_to_end_dist)
    radius_of_gyration = analyzer.compute_radius_of_gyration()
    print(radius_of_gyration.shape)
    results = analyzer.compute_gyration_tensor_properties()
    eigenvalues = np.sqrt(np.array(results["eigenvalues"]).sum(axis=1))
    print(eigenvalues.shape)
    print("Diff:", np.sum((eigenvalues - radius_of_gyration) ** 2))


if __name__ == "__main__":
    test_IDP_path = "data/IDRome/IDRome_v4/Q5/SY/C1/270_327/"
    converter = IDRomeBackmapper(test_IDP_path + "top.pdb", test_IDP_path + "traj.xtc")
    converter._fix_topology(
        out_pdb=test_IDP_path + "top_fixed.pdb",
        out_dcd=test_IDP_path + "traj_fixed.dcd",
    )
    traj = md.load(
        test_IDP_path + "traj_fixed.dcd", top=test_IDP_path + "top_fixed.pdb"
    )
    traj.save_xtc(test_IDP_path + "traj_fixed.xtc")

    compute_basic_prop(test_IDP_path, suffix="_fixed")
    compute_basic_prop(test_IDP_path, suffix="_AA")
