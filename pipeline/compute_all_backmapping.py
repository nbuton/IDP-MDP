import logging
from idpmdp.backmapping import IDRomeBackmapper
from idpmdp.utils import get_pdb_directories


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
