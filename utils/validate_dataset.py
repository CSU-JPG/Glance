import os
import argparse
from pathlib import Path

def validate_dataset(data_path):
    """
    Validate that each image in the dataset directory has a corresponding text file and vice versa.

    Args:
        data_path (str or Path): Path to the dataset directory.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """
    data_dir = Path(data_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    images = []
    texts = []

    for file in data_dir.iterdir():
        if file.suffix.lower() in image_extensions:
            images.append(file.stem)
        elif file.suffix == '.txt':
            texts.append(file.stem)

    images.sort()
    texts.sort()

    print(f"Found {len(images)} images and {len(texts)} text files")

    missing_texts = set(images) - set(texts)
    missing_images = set(texts) - set(images)

    if missing_texts:
        print(f"Missing text files for: {missing_texts}")
    if missing_images:
        print(f"Missing images for: {missing_images}")

    if not missing_texts and not missing_images:
        print("Dataset structure is valid!")

    return len(images) == len(texts) and not missing_texts and not missing_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate image-text dataset structure.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset directory to validate."
    )
    args = parser.parse_args()
    validate_dataset(args.path)
