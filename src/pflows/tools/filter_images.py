import os
import random
import re

from typing import List

from pflows.typedef import Dataset, Image


def sample_group_proportion(dataset: Dataset, number: int, seed: int = 49) -> Dataset:
    shuffled_images = dataset.images
    random.seed(seed)
    random.shuffle(shuffled_images)
    groups_numbers = {}
    for image in shuffled_images:
        if image.group not in groups_numbers:
            groups_numbers[image.group] = 0
        groups_numbers[image.group] += 1
    # calculate the number of images to take per group
    for group in list(groups_numbers.keys()):
        groups_numbers[group] = int(number * (groups_numbers[group] / len(shuffled_images)))
    if sum(groups_numbers.values()) < number:
        # if the sum of the number of images is less than the number
        # we will add the remaining images to the group with the most images
        remaining = number - sum(groups_numbers.values())
        max_group = max(groups_numbers, key=groups_numbers.get)  # type: ignore
        groups_numbers[max_group] += remaining
    # take the images
    images = []
    for group in list(groups_numbers.keys()):
        images += [image for image in shuffled_images if image.group == group][
            : groups_numbers[group]
        ]

    return Dataset(
        images=images,
        categories=dataset.categories,
        groups=dataset.groups,
    )


def sample(
    dataset: Dataset, number: int, offset: int = 0, seed: int = 49, sort: str | None = None
) -> Dataset:
    if dataset.groups is not None and len(dataset.groups) > 1:
        if offset is not None and offset > 0:
            raise ValueError("Offset is not supported when groups are present")
        return sample_group_proportion(dataset, number, seed)

    sorted_images = dataset.images
    new_ids_order = [image.id for image in sorted_images]
    if sort is not None:
        new_ids_order = sorted(new_ids_order)
    else:
        # shuffle the images using seed
        random.seed(seed)
        random.shuffle(new_ids_order)

    new_images = []
    for new_id in new_ids_order:
        image = [image for image in sorted_images if image.id == new_id][0]
        new_images.append(image)
    return Dataset(
        images=new_images[offset : offset + number],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def by_ids(dataset: Dataset, ids: list[str]) -> Dataset:
    return Dataset(
        images=[image for image in dataset.images if image.id in ids],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def name_duplicate(dataset: Dataset, regexp: str) -> Dataset:
    # convert to regexp
    name_regexp_to_compare = re.compile(regexp)
    # We want to check for duplicates in the regexp name
    name_groups: dict[str, List[Image]] = {}
    for image in dataset.images:
        image_name = os.path.basename(image.path)
        match = re.match(name_regexp_to_compare, image_name)
        if match:
            name = match.group(0)
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(image)
    exclude_images = []
    for name, group_images in name_groups.items():
        if len(group_images) > 1:
            # We need to exclude all the other paths
            exclude_images += [image.path for index, image in enumerate(group_images) if index != 0]
    print(f"found {len(exclude_images)} duplicate images")
    dataset.images = [image for image in dataset.images if image.path not in exclude_images]
    return dataset


def by_group(dataset: Dataset, group: str) -> Dataset:
    return Dataset(
        images=[image for image in dataset.images if group == image.group],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def by_groups(dataset: Dataset, groups: List[str]) -> Dataset:
    return Dataset(
        images=[image for image in dataset.images if image.group in groups],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def by_category_name(
    dataset: Dataset, include: List[str] | None = None, exclude: List[str] | None = None
) -> Dataset:
    include = include or []
    exclude = exclude or []
    filtered_images = []

    for image in dataset.images:
        category_names = set(ann.category_name for ann in image.annotations)

        if exclude and any(cat in exclude for cat in category_names):
            continue

        if not include or any(cat in include for cat in category_names):
            filtered_images.append(image)

    return Dataset(images=filtered_images, categories=dataset.categories, groups=dataset.groups)
