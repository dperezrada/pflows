from pflow.typedef import Dataset


def count_images(dataset: Dataset) -> None:
    print()
    print("total images: ", len(dataset.images))


def count_categories(dataset: Dataset) -> None:
    print()
    print("total categories: ", len(dataset.categories))


def show_categories(dataset: Dataset) -> None:
    print()
    print("Categories:")
    for category in dataset.categories:
        print("\t", category.name)
