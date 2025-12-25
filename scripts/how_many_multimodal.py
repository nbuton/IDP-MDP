"""
Count the number of multimodal Rg distributions in the ped data directory.
"""

import os
from idpmdp.utils import test_multimodality
from idpmdp.protein_analyzer import ProteinAnalyzer
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = Path("data/PED")
all_directory = os.listdir(data_dir)

count_multimodal = 0
count_unimodal = 0
for directory in tqdm(all_directory):
    all_pdb_files = os.listdir(data_dir / directory)
    complete_pdb_files = [file for file in all_pdb_files if file.endswith(".pdb")]
    complete_pdb_files = [
        os.path.join(data_dir / directory, file) for file in complete_pdb_files
    ]
    try:
        analyzer = ProteinAnalyzer(complete_pdb_files)
        all_re_values = analyzer.compute_end_to_end_distance()
    except:
        print(f"Failed to analyze {directory}")
        continue

    stat, p, is_multimodal = test_multimodality(all_re_values)
    if is_multimodal:
        count_multimodal += 1
        plt.hist(all_re_values)
        plt.show()
    else:
        count_unimodal += 1

print(f"Multimodal Rg distributions: {count_multimodal}")
print(f"Unimodal Rg distributions: {count_unimodal}")
