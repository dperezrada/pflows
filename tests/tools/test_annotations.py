import pytest
from pflows.tools.annotations import filter_by_tag


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


    
