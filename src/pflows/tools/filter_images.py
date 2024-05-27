import os
import re

from typing import List

from pflows.typedef import Dataset, Image


def sample(dataset: Dataset, number: int, offset: int = 0, sort: str | None = None) -> Dataset:
    sorted_images = dataset.images
    if sort is not None:
        sorted_images = sorted(dataset.images, key=lambda image: getattr(image, sort))
    return Dataset(
        images=sorted_images[offset : offset + number],
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
