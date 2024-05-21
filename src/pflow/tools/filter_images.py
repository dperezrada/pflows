from pflow.typedef import Dataset


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
