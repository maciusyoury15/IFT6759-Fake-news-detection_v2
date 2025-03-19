import os
import gc
import tarfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt


def fetch_images_data(image_tar_path = '/content/drive/MyDrive/Fake_News_Detection/public_images.tar.bz2',
                      image_folder = '/content/drive/MyDrive/Fake_News_Detection/extracted_images'):
    """
    **************** (To run in google colab) ***********
    Downloads a .tar.bz2 file from your Google Drive, extracts it to image_folder which is returned:
    (You should have images .tar.bz2 file in your Google Drive folder)

    Parameters:
    - image_tar_path (str): Google Drive link to the images. (Adjust it to your Google Drive folder)
    - image_folder (str): Path to store extracted images. (Adjust it according to your Google Drive folder)

    Returns:
    - image_folder (str): Path of the extracted images.
    """

    # Create a directory for extracted files
    os.makedirs(image_folder, exist_ok=True)

    # Process the tar file in streaming mode
    extracted_count = 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    try:
        with tarfile.open(image_tar_path, 'r:bz2') as tar:
            for i, member in enumerate(tar):
                # Skip directories and non-image files
                if not member.isfile() or not member.name.lower().endswith(image_extensions):
                    continue

                # Extract file directly to disk
                try:
                    tar.extract(member, path=image_folder)
                    extracted_count += 1
                except Exception as e:
                    print(f"Error extracting {member.name}: {e}")

                # Clean up memory and report progress
                if i % 100 == 0:
                    gc.collect()
                    print(f"Processed {i} files, extracted {extracted_count} images")

                # Check disk space every 1000 files
                if i % 1000 == 0:
                    # !df - h / content  # Code for google colab

                    # Get disk usage statistics for the directory
                    total, used, free = shutil.disk_usage("/content")

                    # Convert to human-readable format (GB)
                    total_gb, used_gb, free_gb = total // (2 ** 30), used // (2 ** 30), free // (2 ** 30)

                    print(f"Disk space: {free_gb} GB free out of {total_gb} GB total ({used_gb} GB used)")

    except Exception as e:
        print(f"Error processing archive: {e}")

    print(f"Extraction complete. Total images extracted: {extracted_count}, to {image_folder}")

    return image_folder


def display_sample_images(folder_path = '/content/drive/MyDrive/Fake_News_Detection/extracted_images/public_image_set', num_samples=5):
    """
    Displays sample images from the image folder.

    Parameters:
    - image_folder (str): Path to store extracted images. (Adjust it according to your Google Drive folder)
    - num_samples (int): Number of images to display.
    """

    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # Display the first 5 images
    for i, image_file in enumerate(image_files[:num_samples]):
        image_path = os.path.join(folder_path, image_file)

        try:
            # Open and display the image
            img = Image.open(image_path)
            plt.figure(figsize=(10, 10))
            plt.title(f"Image {i + 1}: {image_file}")
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error opening image {image_file}: {e}")