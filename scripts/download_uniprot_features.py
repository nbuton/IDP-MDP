import requests
import json
import time
import os
import re
from tqdm import tqdm


def get_unique_uniprot_ids(root_folder):
    """
    Traverses IDRome nested folders (e.g., A5/A3/E0/) to reconstruct
    and collect unique UniProt IDs.
    """
    unique_ids = set()

    # This regex looks for 3 levels of 2-character alphanumeric folders
    # e.g., .../A5/A3/E0/...
    path_pattern = re.compile(r"([A-Z0-9]{2})[\\/]([A-Z0-9]{2})[\\/]([A-Z0-9]{2})")

    print(f"Scanning directory: {root_folder}...")

    for root, dirs, files in os.walk(root_folder):
        # We check if the current folder path contains the ID structure
        match = path_pattern.search(root)
        if match:
            # Reconstruct the ID: A5 + A3 + E0 = A5A3E0
            uniprot_id = "".join(match.groups())
            unique_ids.add(uniprot_id)

    print(f"Found {len(unique_ids)} unique UniProt IDs.")
    return list(unique_ids)


def get_sequence_features(uniprot_ids, output_path):
    # Switch to 'search' which handles complex boolean queries better than 'stream'
    api_url = "https://rest.uniprot.org/uniprotkb/search"

    # 25-50 is the safe zone for 'search' endpoint boolean queries
    chunk_size = 40

    print(f"Fetching features for {len(uniprot_ids)} IDs using Search API...")

    with open(output_path, "w") as f:
        # Iterate through chunks
        for i in tqdm(range(0, len(uniprot_ids), chunk_size), desc="Downloading"):
            chunk = uniprot_ids[i : i + chunk_size]

            # Construct query: accession:P123 OR accession:Q456...
            query = " OR ".join([f"accession:{acc}" for acc in chunk])
            print(query)

            # Parameters for the search endpoint
            # size=50 ensures we get all results for this chunk in one page
            params = {
                "query": query,
                "fields": [
                    "accession",
                    "sequence",
                    "ft_act_site",
                    "ft_binding",
                    "ft_dna_bind",
                    "ft_site",
                ],
                "sort": "accession desc",
                "size": str(chunk_size),
            }
            headers = {"accept": "application/json"}

            # Retry loop for stability
            success = False
            for attempt in range(3):
                try:
                    # POST is safer for long query strings
                    response = requests.get(api_url, headers=headers, params=params)
                    response.raise_for_status()

                    data = response.json()
                    results = data["results"]
                    print(len(results))

                    for entry in results:
                        f.write(json.dumps(entry) + "\n")

                    success = True
                    break  # Break retry loop on success

                except requests.exceptions.RequestException as e:
                    time.sleep(2 * (attempt + 1))  # Backoff: 2s, 4s, 6s

            if not success:
                print(
                    f"\nFailed to fetch batch starting at {i}. IDs might be invalid or server busy."
                )

    print(f"Done. Saved to {output_path}")


# --- EXECUTION ---
if __name__ == "__main__":
    my_ids = get_unique_uniprot_ids("data/IDRome/IDRome_v4/")
    print(my_ids[:10])
    output_file = "data/idrome_uniprot_features.jsonl"

    get_sequence_features(my_ids, output_file)
