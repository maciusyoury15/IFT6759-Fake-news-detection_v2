import os
import pandas as pd
import gdown

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

# if __name__ == "__main__":
#     datasets = fetch_fakeddit_data()