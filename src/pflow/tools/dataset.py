import math
import random
from typing import List, Tuple
from pflow.typedef import Dataset, Image


def re_split_dataset(
    dataset: Dataset,
    train_percentage: float = 0.7,
    val_percentage: float = 0.2,
    test_percentage: float = 0.1,
) -> Dataset:
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
    return Dataset(images=images, categories=dataset.categories, groups=groups)


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
