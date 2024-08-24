from hashlib import md5
from dataclasses import asdict
from typing import Any, Sequence

from pflows.polygons import (
    bbox_from_polygon,
    calculate_center_from_polygon,
    merge_polygons,
    polygon_from_bbox,
)
from pflows.typedef import Annotation, Dataset, Category, Image


def keep(dataset: Dataset, categories: list[str]) -> Dataset:
    """
    Filters a dataset by keeping only the specified categories.

    This function takes a dataset and a list of category names and returns a new dataset
    containing only the images and annotations that belong to the specified categories.
    It iterates over each image in the dataset and filters its annotations based on the
    provided categories. If an image has at least one annotation belonging to the specified
    categories, it is included in the new dataset along with its filtered annotations.

    Args:
        dataset: The input dataset to be filtered.
        categories: A list of category names to keep in the filtered dataset.

    Returns:
        A new dataset containing only the images and annotations belonging to the
        specified categories.

    Example:
        # Define the categories to keep
        categories_to_keep = [
            "11", "12", "13", "14", "15", "16", "17", "18"
        ]

        # Call the keep function
        filtered_dataset = keep(dataset, categories_to_keep)

        # The filtered_dataset will only contain images and annotations belonging to the
        # specified categories.
        # For example, if an image has annotations with categories "11", "12", and "33", only
        # the annotations with categories "11" and "12" will be kept, while the annotation with
        # category "33" will be removed.
        # If an image has no annotations belonging to the specified categories, it will be excluded
        # from the filtered dataset.
    """
    new_categories = [category for category in dataset.categories if category.name in categories]
    print()
    print("New categories: ")
    for category in new_categories:
        print("\t", category.name)
    return remap_category_ids(
        Dataset(
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
                if any(annotation.category_name in categories for annotation in image.annotations)
            ],
            categories=new_categories,
            groups=dataset.groups,
        )
    )


def group_categories(
    dataset: Dataset, groups: dict[str, Sequence[Sequence[str]]], condition: str = "any"
) -> Dataset:
    """
    Groups annotations in a dataset based on the provided category groups.

    Args:
        dataset: The input dataset containing images and annotations.
        groups: A dictionary specifying the category groups. The keys represent the new category
                names, and the values are lists of lists, where each inner list contains the
                categories to be grouped together.
        condition: "any" or "all". If "any", the annotations will be grouped if they have at least
                one category in common. If "all", the annotations will be grouped only if they
                have all categories in common. Default is "any".

    Returns:
        A new dataset with the annotations grouped according to the specified category groups.

    Raises:
        ValueError: If the annotations being grouped have a task other than "detect".

    Example:
        # Define the category groups
        groups = {
            "11_31": [["11", "31"], ["21", "41"]],
            "12_32": [["12", "32"], ["22", "42"]],
            "11": [["11"]],
            "12": [["12"]],
            ...
        }

        # Call the group_categories function
        new_dataset = group_categories(dataset, groups)

        # The new_dataset will contain annotations grouped according to the specified groups
        # For example, annotations with categories "11" and "31" will be merged into a single
        # annotation with the new category "11_31". Similarly, annotations with categories
        # "12" and "32" will be merged into a single annotation with the new category "12_32".
        # The size of the bbox with be the size of the biggest bbox among the annotations
        # in the group.
    """
    new_images = []
    categories_names = list(groups.keys())
    for image in dataset.images:
        new_annotations = []
        all_replace_categories = [el for rows in groups.values() for row in rows for el in row]
        task = [
            ann.task for ann in image.annotations if ann.category_name in all_replace_categories
        ][0]
        for new_category, replace_category_groups in groups.items():
            polygon = None
            for replace_category_group in replace_category_groups:
                annotations = [
                    annotation
                    for annotation in image.annotations
                    if annotation.category_name in replace_category_group
                ]
                if len(annotations) == 0:
                    continue

                if condition == "all" and len(
                    list(set(annotation.category_name for annotation in annotations))
                ) != len(replace_category_group):
                    continue

                annotations_polygons = [
                    annotation.segmentation
                    for annotation in annotations
                    if annotation.segmentation is not None
                ]
                if len(annotations_polygons) == 0:
                    annotations_polygons = [
                        polygon_from_bbox(annotation.bbox)
                        for annotation in annotations
                        if annotation.bbox is not None
                    ]
                    if len(annotations_polygons) == 0:
                        continue
                new_polygon = merge_polygons(annotations_polygons)
                polygon = merge_polygons([p for p in [polygon, new_polygon] if p is not None])

            if polygon is None:
                continue
            bbox = bbox_from_polygon(polygon)
            new_annotations.append(
                Annotation(
                    id=md5(
                        "_".join([str(annotation.id) for annotation in annotations]).encode()
                    ).hexdigest(),
                    category_id=categories_names.index(new_category),
                    center=calculate_center_from_polygon(polygon),
                    bbox=bbox,
                    segmentation=polygon,
                    task=image.annotations[0].task,
                    conf=(
                        round(
                            sum(annotation.conf for annotation in annotations) / len(annotations),
                            3,
                        )
                        if len(annotations) > 0
                        else -1.0
                    ),
                    category_name=new_category,
                    tags=list(set(tag for annotation in annotations for tag in annotation.tags)),
                )
            )

        new_images.append(
            Image(
                **{
                    **asdict(image),
                    "annotations": new_annotations,
                }
            )
        )
    return remap_category_ids(
        Dataset(
            images=new_images,
            categories=[
                Category(
                    id=index,
                    name=name,
                )
                for index, name in enumerate(categories_names)
            ],
            groups=dataset.groups,
        )
    )


def remap_category_ids(
    dataset: Dataset, check_missing_categories=True, dataclasses: dict[str, Any] | None = None
) -> Dataset:
    """
    Remaps the category ids in a dataset to be contiguous integers starting from 0.
    """
    if dataclasses is None:
        dataclasses = {}

    Image_class = dataclasses.get("Image", Image)
    Annotation_class = dataclasses.get("Annotation", Annotation)
    Category_class = dataclasses.get("Category", Category)
    Dataset_class = dataclasses.get("Dataset", Dataset)

    category_names = sorted(list(set(category.name for category in dataset.categories)))
    if check_missing_categories:
        category_names = sorted(
            list(
                set(
                    annotation.category_name
                    for image in dataset.images
                    for annotation in image.annotations
                )
            )
        )

    new_categories = [
        Category_class(
            id=index,
            name=name,
        )
        for index, name in enumerate(category_names)
    ]

    # Remap the category ids
    category_map = {category.name: category.id for category in new_categories}
    new_images = [
        Image_class(
            **{
                **asdict(image),
                "annotations": [
                    Annotation_class(
                        **{
                            **asdict(annotation),
                            "category_id": category_map[annotation.category_name],
                        }
                    )
                    for annotation in image.annotations
                ],
            }
        )
        for image in dataset.images
    ]
    return Dataset_class(
        images=new_images,
        categories=new_categories,
        groups=dataset.groups,
    )


def remap_categories(dataset: Dataset, mapping: dict[str, str]) -> Dataset:
    """
    Renames categories in a dataset based on the provided category map.

    Args:
        dataset: The input dataset containing images and annotations.
        category_map: A dictionary specifying the mapping of old category names to new category names.

    Returns:
        A new dataset with the categories renamed according to the specified category map.

    Example:
        # Define the category map
        category_map = {
            "11": "A",
            "12": "B",
            "13": "C",
            ...
        }

        # Call the rename_categories function
        new_dataset = rename_categories(dataset, category_map)

        # The new_dataset will contain categories renamed according to the specified category map.
        # For example, categories "11", "12", and "13" will be renamed to "A", "B", and "C"
        # respectively.
    """
    new_category_names = list(set(mapping.values()))
    new_categories = [
        Category(
            id=index,
            name=name,
        )
        for index, name in enumerate(new_category_names)
    ]
    return remap_category_ids(
        Dataset(
            images=[
                Image(
                    **{
                        **asdict(image),
                        "annotations": [
                            Annotation(
                                **{
                                    **asdict(annotation),
                                    "category_name": mapping.get(
                                        annotation.category_name, annotation.category_name
                                    ),
                                }
                            )
                            for annotation in image.annotations
                        ],
                    }
                )
                for image in dataset.images
            ],
            categories=new_categories,
            groups=dataset.groups,
        )
    )


def rename_some_categories(dataset: Dataset, renames: dict[str, str]) -> Dataset:
    """
    Renames specified categories in a dataset.

    This function takes a dataset and a dictionary of renames where keys are the original category names
    and values are the new names. It returns a new dataset with updated category names for the specified
    categories. Categories not specified in the renames dictionary remain unchanged.

    Args:
        dataset: The input dataset to be modified.
        renames: A dictionary mapping from old category names to new category names.

    Returns:
        A new dataset with updated category names.

    Example:
        # Define the renames
        renames = {
            "11": "A",
            "12": "B",
            "13": "C"
        }

        # Call the rename_some_categories function
        updated_dataset = rename_some_categories(dataset, renames)

        # The updated_dataset will have categories "11", "12", and "13" renamed to "A", "B", and "C"
        # respectively, while other categories remain unchanged.
    """
    new_images = [
        Image(
            **{
                **asdict(image),
                "annotations": [
                    Annotation(
                        **{
                            **asdict(annotation),
                            "category_name": renames.get(
                                annotation.category_name, annotation.category_name
                            ),
                        }
                    )
                    for annotation in image.annotations
                ],
            }
        )
        for image in dataset.images
    ]
    updated_categories = [
        Category(id=category.id, name=renames.get(category.name, category.name))
        for category in dataset.categories
    ]
    return remap_category_ids(
        Dataset(
            images=new_images,
            categories=updated_categories,
            groups=dataset.groups,
        )
    )
