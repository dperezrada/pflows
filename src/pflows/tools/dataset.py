import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
from pflows.typedef import Annotation, Category, Dataset, Image


# pylint: disable=too-many-locals
def re_split_dataset(
    dataset: Dataset,
    train_percentage: float = 0.7,
    val_percentage: float = 0.2,
    test_percentage: float = 0.1,
) -> Dict[str, Any]:
    # Validate percentages
    total_percentage = train_percentage + val_percentage + test_percentage
    rounded_total = math.ceil(total_percentage * 100)
    if rounded_total != 100:
        raise ValueError("The sum of the percentages must be 1.0")
    # Separate augmented images
    images_to_resplit = [image for image in dataset.images if "augmented" not in (image.tags or [])]
    augmented_images = [image for image in dataset.images if "augmented" in (image.tags or [])]
    random.shuffle(images_to_resplit)

    total_images = len(dataset.images)
    train_total_images, val_total_images, test_total_images = calculate_total_images(
        total_images, train_percentage, val_percentage, test_percentage, len(augmented_images)
    )
    groups_info = [
        ("train", 0, train_total_images),
        ("val", train_total_images, train_total_images + val_total_images),
        ("test", train_total_images + val_total_images, total_images),
    ]
    images = assign_groups(images_to_resplit, augmented_images, groups_info)

    # Determine groups
    groups = []
    if train_total_images > 0:
        groups.append("train")
    if val_total_images > 0:
        groups.append("val")
    if test_total_images > 0:
        groups.append("test")

    images_ids_by_group: Dict[str, List[Dict[str, Any]]] = {group: [] for group in groups}
    for image in images:
        images_ids_by_group[image.group].append(
            {
                "id": image.id,
                "name": Path(image.path).name,
                "intermediate_ids": image.intermediate_ids,
            }
        )
    return {
        "dataset": Dataset(images=images, categories=dataset.categories, groups=groups),
        "ids_by_group": images_ids_by_group,
    }


def calculate_total_images(
    total_images: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    num_augmented_images: int,
) -> Tuple[int, int, int]:
    train_total_images = int(total_images * train_percentage)
    train_total_images_without_augmented = train_total_images - num_augmented_images
    if train_total_images_without_augmented < 0:
        train_total_images = num_augmented_images
    else:
        train_total_images = train_total_images_without_augmented

    remaining_images = total_images - train_total_images
    val_total_images = int(remaining_images * (val_percentage / (val_percentage + test_percentage)))
    test_total_images = remaining_images - val_total_images
    return train_total_images, val_total_images, test_total_images


def assign_groups(
    images_to_resplit: List[Image],
    augmented_images: List[Image],
    groups_info: List[Tuple[str, int, int]],
) -> List[Image]:
    images = []
    for image in augmented_images:
        image.group = "train"
        images.append(image)
    for group_name, images_start, images_end in groups_info:
        for image in images_to_resplit[images_start:images_end]:
            image.group = group_name
            images.append(image)
    return images


def write_json_dataset(dataset: Dataset, target_path=None) -> Dataset:
    to_return = []
    for image in dataset.images:
        to_return.append(
            {
                **image.__dict__,
                "annotations": [annotation.__dict__ for annotation in image.annotations],
            }
        )
    if target_path is None:
        # generate temporary file
        target_path = Path("dataset.json")

    with open(target_path, "w") as file:
        file.write(
            json.dumps(
                {
                    "images": to_return,
                    "categories": [category.__dict__ for category in dataset.categories],
                    "groups": dataset.groups,
                },
                indent=4,
                ensure_ascii=False,
            )
        )
    print("Dataset written to", target_path)

    return dataset


def read_json_dataset(dataset: Dataset | None = None, target_path=None) -> Dataset:
    if target_path is None:
        # generate temporary file
        target_path = Path("dataset.json")

    with open(target_path, "r") as file:
        data = json.load(file)
        images = []
        for image in data["images"]:
            annotations = []
            for annotation in image["annotations"]:
                annotations.append(Annotation(**annotation))
            images.append(Image(**{**image, "annotations": annotations}))  # type: ignore
        categories = []
        for category in data["categories"]:
            categories.append(Category(**category))
        return Dataset(images=images, categories=categories, groups=data["groups"])

    return dataset


def reset_dataset(dataset: Dataset) -> Dataset:
    return Dataset(
        images=[],
        categories=[],
        groups=[],
    )
