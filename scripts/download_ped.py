from idpmdp.downloader.ped_downloader import PEDDownloader
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ped_downloader = PEDDownloader()
    ped_downloader.download_all(output_root="data/PED", max_nb_ids=None)
