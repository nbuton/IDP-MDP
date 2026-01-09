from idpmdp.dataset_creator.split_and_create_datasets import (
    create_all_sequences_csv,
    get_asymmetric_splits_from_csv,
    unify_selected_idrome_to_h5,
)

if __name__ == "__main__":
    root_directory = "data/IDRome/IDRome_v4/"
    path_output_all_sequences_csv = "data/all_sequences.csv"
    if not os.path.exists(path_output_all_sequences_csv):
        create_all_sequences_csv(
            root_directory, output_csv=path_output_all_sequences_csv
        )

    dataset_splits = get_asymmetric_splits_from_csv(
        path_output_all_sequences_csv,
        min_id=0.4,
        cov_ratio=0.8,  # Ensure 80% of the sequence length is covered
        cov_mode=0,  # Coverage of both query and target
        cluster_mode=0,  # Set-cover (most representative clusters)
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    print(f"Train: {len(dataset_splits['train'])}")
    print(f"Val: {len(dataset_splits['validation'])}")
    print(f"Test: {len(dataset_splits['test'])}")

    path_final_dataset = "data/DynaFoldBench/"
    os.makedirs(path_final_dataset, exist_ok=True)
    for split in ["train", "validation", "test"]:
        unify_selected_idrome_to_h5(
            root_directory,
            path_output_h5=path_final_dataset + split + ".h5",
            selected_entries=dataset_splits[split],
        )
