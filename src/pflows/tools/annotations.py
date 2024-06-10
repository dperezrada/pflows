from typing import List
from dataclasses import asdict

from pflows.typedef import Dataset, Image


def filter_by_tag(
    dataset: Dataset, include: List[str] | None = None, exclude: List[str] | None = None
) -> Dataset:
    include = include or []
    exclude = exclude or []

    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        annotation
                        for annotation in image.annotations
                        if (len(include) == 0 or any(tag in (annotation.tags or []) for tag in include))
                        and (len(exclude) == 0 or all(tag not in (annotation.tags or []) for tag in exclude))
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=dataset.categories,
        groups=dataset.groups,
    )

def keep_certain_categories(dataset: Dataset, categories: List[str]) -> Dataset:
    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        annotation
                        for annotation in image.annotations
                        if annotation.category_name in categories
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=dataset.categories,
        groups=dataset.groups,
    )