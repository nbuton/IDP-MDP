from idpmdp.downloader.ped_downloader import PEDDownloader
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ped_downloader = PEDDownloader()
    ped_downloader.download_all_experimental(
        output_root="data/PED",
        experimental_terms=["SAXS", "NMR", "FRET", "SANS", "BMRB", "SASBDB"],
        max_nb_ids=None,
    )
