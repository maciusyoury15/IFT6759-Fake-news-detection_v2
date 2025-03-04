import os
import pandas as pd
import numpy as np
import gdown

import requests
from io import BytesIO
from PIL import Image

def fetch_fakeddit_data(output_dir="data/raw"):
    """
    Fetches all Fakeddit datasets (train, validation, and test) and saves them in a folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Fakeddit dataset file IDs (these are the actual Google Drive file IDs)
    file_ids = {
        "train": "13nQ5bAFYfgqlDJErtzFXhfg95gvj00Sp",  # train.tsv
        "validate": "1CbiY2tC54sqr95T9CUljD6B4ZwVcdPEK",  # validate.tsv
        "test": "1yMOilSR3UVAfiV_pGYw05cocHx6XAuaq"  # test.tsv
    }

    datasets = []
    for set_name, file_id in file_ids.items():
        file_path = os.path.join(output_dir, f"{set_name}_raw.tsv").replace("\\", "/")
        url = f"https://drive.google.com/uc?id={file_id}"

        if not os.path.isfile(file_path):
            print(f"Downloading {set_name} dataset...")
            gdown.download(url, file_path, quiet=False)
            print(f"{set_name} dataset downloaded to {file_path}")
        else:
            print(f"{set_name} dataset already exists: {file_path}")


        # Load the data
        try:
            data = pd.read_csv(file_path, sep='\t')
            datasets.append(data)
            print(f"Loaded {set_name} dataset with {len(data)} records")
        except Exception as e:
            print(f"Error loading {set_name} dataset: {e}")
            # If there's an error, check the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = ''.join([f.readline() for _ in range(5)])
                print(f"First few lines of the file:\n{first_lines}")

    return datasets



def fetch_image_from_url(url):
    """
    Fetches an image from a URL and returns a PIL image object.

    Parameters:
    - url (str or NaN): The image URL.

    Returns:
    - PIL Image object if successful, otherwise None.
    """
    # Custom headers to mimic a real browser
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    
    # Skip NaN or invalid URL values
    if pd.isna(url) or str(url).strip().lower() in ["nan", ""]:
        print(f"Skipping invalid URL: {url}")
        return None

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)

        # Check HTTP response
        if response.status_code != 200:
            print(f"Failed to fetch image (Status {response.status_code}): {url}")
            return None

        # Check first bytes of response
        content_type = response.headers.get("Content-Type", "")
        print(f"Fetched URL: {url}, Content-Type: {content_type}")

        # Ensure it's an image
        if "image" not in content_type:
            print(f"URL is not an image: {url}")
            return None

        # Open and return the image
        return Image.open(BytesIO(response.content))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
        return None



# if __name__ == "__main__":
#     datasets = fetch_fakeddit_data()