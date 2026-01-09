import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    h5_path = "data/unified_dataset.h5"
    property_name = "asphericity_std"
    all_asphericity = []
    with h5py.File(h5_path, "r") as source_f:
        for key, obj in source_f.items():  # Browse all IDPs
            sequence = obj.attrs["sequence"]
            print(sequence)
            asphericity_mean = np.array(obj[property_name])
            all_asphericity.append(asphericity_mean)

    plt.hist(all_asphericity)
    plt.savefig("distribution_" + property_name + ".png")
