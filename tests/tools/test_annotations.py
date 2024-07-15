import pytest
from pflows.tools.annotations import change_all_categories, filter_by_tag


def test_filter_by_tag(dataset):
    assert len(dataset.images) == 8
    total_annotations = sum(len(image.annotations) for image in dataset.images)
    dataset.images[0].annotations[0].tags = ["tag1"]
    dataset.images[1].annotations[0].tags = ["tag2"]
    dataset.images[2].annotations[0].tags = ["tag1", "tag2"]
    dataset.images[3].annotations[0].tags = ["tag3"]
    
    expected_annotations = total_annotations - 2
    assert sum(len(image.annotations) for image in filter_by_tag(dataset, exclude=["tag1"]).images) == expected_annotations

    expected_annotations = total_annotations - 2
    assert sum(len(image.annotations) for image in filter_by_tag(dataset, exclude=["tag2"]).images) == expected_annotations

    expected_annotations = total_annotations - 3
    assert sum(len(image.annotations) for image in filter_by_tag(dataset, exclude=["tag2", "tag3"]).images) == expected_annotations

    expected_annotations = 1
    assert sum(len(image.annotations) for image in filter_by_tag(dataset, include=["tag3"]).images) == expected_annotations


    
def test_replace_all_categories(dataset):
    assert len(dataset.images) == 8
    total_annotations = sum(len(image.annotations) for image in dataset.images)
    total_annotation_categories = len(set([
        annotation.category_name for image in dataset.images for annotation in image.annotations
    ]))
    assert total_annotation_categories == 10
    new_dataset = change_all_categories(dataset, "new_category")
    assert len(new_dataset.images) == 8
    total_annotation_categories_new = len(set([
        annotation.category_name for image in new_dataset.images for annotation in image.annotations
    ]))
    assert total_annotation_categories_new == 1
    total_annotation_categories_new_ids = len(set([
        annotation.category_id for image in new_dataset.images for annotation in image.annotations
    ]))
    assert total_annotation_categories_new_ids == 1
