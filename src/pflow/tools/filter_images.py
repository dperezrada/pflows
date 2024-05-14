from pflow.typedef import Dataset, DatasetDict


def sample(dataset: Dataset, number: int, offset: int = 0) -> DatasetDict:
    return {
        "dataset": Dataset(
            images=dataset.images[offset : offset + number],
            categories=dataset.categories,
        )
    }


def by_ids(dataset: Dataset, ids: list[str]) -> DatasetDict:
    return {
        "dataset": Dataset(
            images=[image for image in dataset.images if image.id in ids],
            categories=dataset.categories,
        )
    }
