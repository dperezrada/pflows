import os
import time
import sys
import glob
from typing import Any, Dict, List
from pathlib import Path
import shutil
import zipfile


from pflows.model import get_image_info
from pflows.typedef import Dataset, Image

ALLOWED_IMAGES = [".jpg", ".png", ".jpeg"]


def count_images(dataset: Dataset) -> Dict[str, int]:
    print()
    print("total images: ", len(dataset.images))
    return {"count": len(dataset.images)}


def count_categories(dataset: Dataset) -> Dict[str, Any]:
    print()
    print("total categories: ", len(dataset.categories))
    # We count from annotations
    total_categories = {}
    categories_names = {}
    for image in dataset.images:
        for annotation in image.annotations:
            if annotation.category_id in total_categories:
                total_categories[annotation.category_id] += 1
            else:
                categories_names[annotation.category_id] = annotation.category_name
                total_categories[annotation.category_id] = 1
    print("Total Categories in annotations: ", len(total_categories))
    for category_id, total in total_categories.items():
        print("\t", category_id, categories_names[category_id], ":", total)
    return {"count": len(dataset.categories), "total_categories": total_categories}


def show_categories(dataset: Dataset) -> None:
    print()
    print("Categories:")
    for category in dataset.categories:
        print("\t", category.name)


def count_groups(dataset: Dataset) -> Dict[str, Any]:
    print()
    print("total groups: ", len(dataset.groups))
    groups_total = {}
    for group in dataset.groups:
        groups_total[group] = len([image for image in dataset.images if image.group == group])
        print("\t", group, ":", groups_total[group])
    return {"count": len(dataset.groups), "groups_total": groups_total}


def check_folder(folder: str) -> None:
    # We check if the folder exists and if its a folder
    if not os.path.exists(folder):
        raise FileNotFoundError("The folder does not exist")
    if not os.path.isdir(folder):
        raise NotADirectoryError("The specified path is not a directory")


def find_images_recursively(base_path: str) -> List[str]:
    # Patterns to match
    file_patterns = [f"*{ext}" for ext in ALLOWED_IMAGES]

    # List to hold all found file paths
    found_files = []

    # Recursively search for files
    for pattern in file_patterns:
        found_files.extend(glob.glob(os.path.join(base_path, "**", pattern), recursive=True))

    return found_files


def search_images_in_folder(folder: str, recursive: bool = False) -> List[str]:
    check_folder(folder)
    # We get all the images in the folder
    base_folder = Path(folder).resolve()
    images_paths = []
    if recursive:
        images_paths = find_images_recursively(str(base_folder))
    else:
        images_paths = [str((base_folder / file).resolve()) for file in os.listdir(base_folder)]
    return images_paths


def read_images_from_folder(folder: str, recursive: bool = False) -> List[Image]:
    check_folder(folder)
    # We get all the images in the folder
    base_folder = Path(folder).resolve()
    images: List[Image] = []
    images_paths = search_images_in_folder(folder, recursive=recursive)
    for image_path in images_paths:
        image_info = get_image_info(image_path, "train")
        images.append(image_info)
    return images


def load_images(
    dataset: Dataset,
    path: str,
    paths: List[str] | None = None,
    recursive: bool = False,
    number: int | None = None,
) -> Dataset:
    # we are going to load the images from the folder
    print()
    paths = paths or [path]
    print("loading images from:", paths)
    print("recursive:", recursive)

    image_paths = []
    if number is not None:
        print("number of images to load:", number)
    for folder_path in paths:
        if not os.path.exists(folder_path):
            continue
        if os.path.isfile(folder_path):
            image_paths.append(folder_path)
            continue
        image_paths += search_images_in_folder(folder_path, recursive=recursive)

    images: List[Image] = []

    if number is not None:
        image_paths = image_paths[:number]

    for image_path in image_paths:
        image_info = get_image_info(image_path, "train")
        images.append(image_info)

    print("total images loaded:", len(images))

    # remove duplicates ids
    already_seen = set()
    total_images = len(images)

    new_images = []
    for image in images:
        if image.id in already_seen:
            print("duplicated id", image.id)
            continue
        already_seen.add(image.id)
        new_images.append(image)

    print("removed duplicates ids on load", total_images - len(images))
    groups = ["train"]
    dataset.images += new_images
    if "train" not in dataset.groups:
        dataset.groups += groups
    return dataset


def terminate() -> None:
    sys.exit(0)


def wait(minutes: int | None = None, seconds: int | None = None) -> None:
    if minutes is not None and seconds is not None:
        raise ValueError("You can only specify one of the arguments")
    total_seconds = 0
    if minutes is not None:
        total_seconds += minutes * 60
    if seconds is not None:
        total_seconds += seconds
    print(f"Waiting for {total_seconds} seconds")
    time.sleep(total_seconds)


def echo(text: str, path: str = "") -> Dict[str, str]:
    if path:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
    else:
        print(text)
    return {"echo": text}


def compress_folder(compress_path: str, output: str) -> Dict[str, str]:
    if not os.path.exists(compress_path):
        raise FileNotFoundError("The folder does not exist")
    clean_output = output
    if output.endswith(".zip"):
        clean_output = output[:-4]
    zip_path = f"{clean_output}.zip"
    if os.path.exists(zip_path):
        # remove the existing zip file
        os.remove(zip_path)
    shutil.make_archive(clean_output, "zip", compress_path)
    return {"status": "compressed", "compress_path": compress_path, "output": output}


def decompress_zip(zip_path: str, output: str) -> Dict[str, str]:
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    print("decompressing zip file", zip_path, "to", output)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output)
    return {"status": "uncompressed", "zip_path": zip_path, "output": output}


def print_sample(dataset: Dataset) -> None:
    print("Dataset Info:")
    print("Total images:", len(dataset.images))
    print("Total categories:", len(dataset.categories))
    print("Total groups:", len(dataset.groups))
    print("sample", dataset.images[0])
