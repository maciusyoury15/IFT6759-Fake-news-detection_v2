import os
import gdown
import tarfile
from PIL import Image
import matplotlib.pyplot as plt


def fetch_images_data(output_folder="data/raw/images"):
    """
    Downloads a .tar.bz2 file from Google Drive and extracts it to "data/raw/images"

    Parameters:
    - link_images (str): Google Drive link to the images.
    - output_folder (str): Path to store extracted images.

    Returns:
    - extracted_path (str): Path to the extracted images.
    """

    link_images = "https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view"
    # Convert Google Drive link to a direct download link
    file_id = link_images.split("/d/")[1].split("/")[0]
    direct_link = f"https://drive.google.com/uc?id={file_id}"

    # Define local tar file name
    tar_bz2_filename = "images_dataset.tar.bz2"

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Download the file
    print("Downloading file...")
    gdown.download(direct_link, tar_bz2_filename, quiet=False)

    # Extract the files
    print("Extracting files...")
    with tarfile.open(tar_bz2_filename, "r:bz2") as tar:
        tar.extractall(path=output_folder)

    # Remove the compressed file to save space
    os.remove(tar_bz2_filename)

    print(f"Extraction complete. Images are stored in: {output_folder}")
    return output_folder


def load_images(folder_path):
    """
    Loads images from 'data/raw/images'.

    Parameters:
    - folder_path (str): Path to the folder containing images.

    Returns:
    - images (dict): Dictionary containing image file names and their PIL image objects.
    """
    images = {}

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            images[file] = Image.open(file_path)
            print(f"Loaded Image: {file}")

    return images


def display_sample_images(images, num_samples=5):
    """
    Displays sample images from the dataset.

    Parameters:
    - images (dict): Dictionary containing image filenames and their PIL objects.
    - num_samples (int): Number of images to display.
    """
    sample_keys = list(images.keys())[:num_samples]
    plt.figure(figsize=(10, 5))

    for i, key in enumerate(sample_keys):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[key])
        plt.title(key)
        plt.axis("off")

    plt.show()



# if __name__ == "__main__":
#     # Example usage:
#     images_extracted = fetch_images_data()
#     images_dataset = load_images(images_extracted)
#
#     # Display a few sample images
#     display_sample_images(images_dataset, num_samples=5)
