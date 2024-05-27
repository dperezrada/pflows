import pytest
from pflows.tools.filter_images import by_ids, sample

def test_sample(dataset):
    assert len(dataset.images) == 8
    current_ids = [image.id for image in dataset.images]
    new_dataset = sample(dataset, number=5, offset=2)
    assert len(new_dataset.images) == 5
    assert current_ids[2] == new_dataset.images[0].id
    assert current_ids[6] == new_dataset.images[4].id

def test_sample_sorted(dataset):
    assert len(dataset.images) == 8
    current_ids = sorted([image.id for image in dataset.images])
    new_dataset = sample(dataset, number=5, offset=2, sort="id")
    assert len(new_dataset.images) == 5
    assert current_ids[2] == new_dataset.images[0].id
    assert current_ids[6] == new_dataset.images[4].id

def test_by_ids(dataset):
    filter_ids = [image.id for image in dataset.images[0:2]]
    new_dataset = by_ids(dataset, filter_ids)
    assert len(new_dataset.images) == 2
    assert new_dataset.images[0].id in filter_ids

# Run the test
pytest.main()