import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_uniformity_metric(weights):
    """
    Calculates a metric A where:
    A = 1 -> Uniform Distribution
    A = 0 -> Dirac (Point-mass) Distribution

    Args:
        weights (list or np.array): Raw values, counts, or probabilities.
                                    (e.g., [10, 10, 10] or [0.1, 0.9])

    Returns:
        float: The metric A (between 0 and 1).
    """
    weights = np.array(weights, dtype=float)

    # 1. Handle edge case: Empty or single-element arrays
    if len(weights) <= 1:
        # If there is only 1 bucket, it is technically 100% certain (Dirac)
        # but also "perfectly uniform" for that single bucket.
        # Mathematically, division by log(1)=0 occurs.
        # Usually, this implies 0 uncertainty.
        return 0.0

    # 2. Normalize raw values to get probabilities (p_i)
    # Add a tiny epsilon to sum to prevent division by zero if all weights are 0
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0

    p = weights / total_weight

    # 3. Filter out zero probabilities to avoid log(0) errors
    # We only care about non-zero p for entropy calculation
    p = p[p > 0]

    # 4. Calculate Shannon Entropy
    # H = - sum(p * log(p))
    entropy = -np.sum(p * np.log(p))

    # 5. Calculate Max Possible Entropy (Uniform Case)
    # H_max = log(N)
    max_entropy = np.log(len(weights))

    # 6. Calculate Metric A
    # A = H / H_max
    A = entropy / max_entropy

    return A


def get_ped_master_df(base_dir):
    base_path = Path(base_dir)
    all_data = []

    # Locate all weight files
    weight_files = list(base_path.rglob("e*_weights.csv"))

    for w_file in weight_files:
        try:
            # 1. Basic Path Info
            ped_id = w_file.parts[-2]
            ensemble_id = w_file.name.split("_")[0]

            # 2. Robust Weight Parsing
            # sep=None with engine='python' tells pandas to guess the delimiter
            w_df = pd.read_csv(w_file, header=None, sep=None, engine="python")
            weights = (
                w_df.iloc[:, 1].values if w_df.shape[1] >= 2 else w_df.iloc[:, 0].values
            )

            is_uniform = np.allclose(weights, weights[0], atol=1e-5)
            weight_sum = np.sum(weights)

            if not is_uniform:
                print(calculate_uniformity_metric(weights))
                plt.hist(weights)
                plt.show()

            # 3. Metadata Integration
            meta_file = w_file.parent / f"{ensemble_id}_full_metadata.csv"

            # Default values if metadata file is missing
            meta_dict = {
                "organism": "Unknown",
                "exp_methods": "N/A",
                "title": "N/A",
                "is_human": False,
            }

            if meta_file.exists():
                m_df = pd.read_csv(meta_file)
                if not m_df.empty:
                    row = m_df.iloc[0]
                    meta_dict["organism"] = str(row.get("organism", "Unknown"))
                    meta_dict["exp_methods"] = str(row.get("exp_methods", "N/A"))
                    meta_dict["title"] = str(row.get("title", "N/A"))
                    meta_dict["is_human"] = (
                        "HOMO SAPIENS" in meta_dict["organism"].upper()
                    )

            # 4. Combine into one record
            entry_record = {
                "PED_ID": ped_id,
                "Ensemble": ensemble_id,
                "Models": len(weights),
                "Status": "Uniform" if is_uniform else "BIASED",
                "Weight_Sum": round(weight_sum, 3),
                "Is_Human": meta_dict["is_human"],
                "Organism": meta_dict["organism"],
                "Methods": meta_dict["exp_methods"],
                "Title": meta_dict["title"],
            }
            all_data.append(entry_record)

        except Exception as e:
            print(f"Skipping {w_file.name} due to error: {e}")

    # Create the DataFrame
    master_df = pd.DataFrame(all_data)

    # Sort for better readability
    if not master_df.empty:
        master_df = master_df.sort_values(["PED_ID", "Ensemble"]).reset_index(drop=True)

    return master_df


# --- EXECUTION ---
df = get_ped_master_df("data/PED")

# Display the result
print("\n--- PED ENSEMBLE MASTER DATAFRAME ---")
print(df)  # to_string() ensures all columns are printed in the terminal
print(df.columns)

# 1. Create the filter
filtered_df = df[
    (df["Methods"].str.contains("NMR") & df["Methods"].str.contains("SAXS"))
].copy()
print(
    f"\n--- Filtered DataFrame (NMR or SAXS) | Found {len(filtered_df)} Ensembles ---"
)
print(filtered_df)

filtered_df = filtered_df[filtered_df["Status"] != "Uniform"].copy()
print(f"\n--- Filtered DataFrame Uniform | Found {len(filtered_df)} Ensembles ---")
print(filtered_df)

# 3. Optional: See a count of each specific type
n_nmr = df["Methods"].str.contains("NMR", case=False, na=False).sum()
n_saxs = df["Methods"].str.contains("SAXS", case=False, na=False).sum()

print(f"\nBreakdown:")
print(f"- Ensembles with NMR: {n_nmr}")
print(f"- Ensembles with SAXS: {n_saxs}")
