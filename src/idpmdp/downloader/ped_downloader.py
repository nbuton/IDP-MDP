"""
Download data from the Protein Ensemble Database (PED) db: https://proteinensemble.org/

This module provides utilities to interact with the PED public API.
"""

import time
from typing import Dict, List, Optional
import requests
import tarfile
import io
import os
import logging
from tqdm import tqdm


class PEDAPIError(RuntimeError):
    """Raised when the PED API returns an unexpected response."""


class PEDDownloader:
    PED_API_BASE = "https://deposition.proteinensemble.org/api/v1/"

    def __init__(self):
        # Do not perform network IO on construction. Call methods explicitly.
        pass

    def list_ped_entry_ids(
        self,
        page_size: int = 100,
        max_nb_ids: Optional[int] = None,
        request_timeout: float = 20.0,
        retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> List[str]:
        """
        Return all entry IDs available in the Protein Ensemble Database (PED).

        Iterates through the paginated PED API and collects the identifier of each
        entry. Resilient to transient network issues via a simple retry/backoff strategy.

        Parameters
        - page_size: Number of items per page requested from the API (max commonly 100).
        - max_nb_ids: Optional upper bound to limit the number of ids fetched (for testing).
        - request_timeout: Per-request timeout in seconds.
        - retries: Number of retry attempts for transient failures per request.
        - backoff_seconds: Initial backoff delay between retries; increases linearly per attempt.

        Returns
        - List[str]: A list of PED entry identifiers.

        Raises
        - PEDAPIError: If the API returns an unexpected schema or a non-OK response after retries.
        """
        logging.info("Getting all PED entry IDs")
        url = f"{self.PED_API_BASE}/entries/"

        ids: List[str] = []
        session = requests.Session()

        offset = 0
        while True:
            logging.info(f"{offset} ids has already been browse")
            # Stop early if we've reached the requested number of IDs
            if max_nb_ids is not None and len(ids) >= max_nb_ids:
                ids = ids[:max_nb_ids]
                break

            params = {"offset": offset, "limit": page_size}

            # Basic retry logic for the page request
            for attempt in range(1, retries + 1):
                try:
                    resp = session.get(url, params=params, timeout=request_timeout)
                    if 500 <= resp.status_code < 600:
                        # server-side transient error -> retry
                        raise PEDAPIError(
                            f"Server error {resp.status_code} on attempt {attempt}"
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (requests.RequestException, ValueError, PEDAPIError) as exc:
                    if attempt == retries:
                        raise PEDAPIError(
                            f"Failed to fetch PED entries page after {retries} attempts: {exc}"
                        )
                    time.sleep(backoff_seconds * attempt)
            else:
                # Should never hit due to raise above, but keeps mypy happy
                raise PEDAPIError("Unreachable retry loop exit")

            results = data.get("result", [])
            if not results:  # End of the results
                break

            for item in results:
                entry_id = item.get("entry_id")
                if entry_id is not None:
                    ids.append(str(entry_id))

            offset += len(results)

        return ids

    def get_ensemble_ids_map(self, ped_ids: List[str]) -> Dict[str, List[str]]:
        """
        For a list of PED IDs, return a map from PED ID -> list of ensemble IDs.
        """
        logging.info("Getting ensemble IDs for PED IDs")
        results_map: Dict[str, List[str]] = {}
        base_url = "https://deposition.proteinensemble.org/api/v1/entries/{}/"

        for ped_id in tqdm(ped_ids, desc="fetching ped ids", unit="ped id"):
            url = base_url.format(ped_id)

            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                data = response.json()

                ensemble_list = data.get("ensembles", [])
                ids = [
                    item["ensemble_id"]
                    for item in ensemble_list
                    if isinstance(item, dict) and "ensemble_id" in item
                ]

                results_map[ped_id] = ids

            except requests.exceptions.RequestException as e:
                print(f"Error fetching {ped_id}: {e}")
                results_map[ped_id] = []

        return results_map

    def download_ped_assets(
        self, ped_id: str, ensemble_id: str, output_dir: str
    ) -> None:
        """
        Downloads specific assets (PDB and weights) for a given PED entry.
        """
        base_url = f"https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/"

        # Define the specific assets you need
        assets = ["ensemble-pdb", "weights"]

        os.makedirs(output_dir, exist_ok=True)

        for asset in assets:
            url = f"{base_url}{asset}"
            try:
                logging.debug(f"Downloading {asset} from: {url}")
                # Stream the download for better memory management with large PDB files
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()

                # Define the local filename (adding .pdb or .csv extension if needed)
                # Some APIs provide the filename in 'Content-Disposition' header
                file_extension = ".pdb" if "pdb" in asset else ".csv"
                file_path = os.path.join(
                    output_dir, f"{ensemble_id}_{asset}{file_extension}"
                )

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.debug(f"Successfully saved to: {file_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {asset}: {e}")

    def download_all(
        self,
        output_root: str = "data/PED",
        page_size: int = 100,
        max_nb_ids: Optional[int] = None,
    ) -> None:
        """
        Fetch all PED IDs, then for each entry fetch ensemble IDs and download
        all ensembles into per-entry folders under output_root.

        Directory structure:
          output_root/
            PEDxxxxx/
              <extracted ensemble files>
        """
        os.makedirs(output_root, exist_ok=True)

        ped_ids = self.list_ped_entry_ids(page_size=page_size, max_nb_ids=max_nb_ids)
        if not ped_ids:
            print("No PED entry IDs found.")
            return

        ped_to_ensembles = self.get_ensemble_ids_map(ped_ids)

        logging.info("Downloading all PED assets")
        for ped_id, ensemble_ids in tqdm(ped_to_ensembles.items()):
            if not ensemble_ids:
                print(f"No ensembles for {ped_id}")
                continue

            ped_dir = os.path.join(output_root, ped_id)
            os.makedirs(ped_dir, exist_ok=True)

            for ensemble_id in ensemble_ids:
                try:
                    logging.debug(f"Downloading {ped_id}/{ensemble_id} -> {ped_dir}")
                    self.download_ped_assets(
                        ped_id=ped_id, ensemble_id=ensemble_id, output_dir=ped_dir
                    )
                except Exception as exc:
                    print(f"Failed {ped_id} {ensemble_id}: {exc}")
