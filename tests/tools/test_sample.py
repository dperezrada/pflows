import pytest
from pflows.tools.filter_images import by_ids, sample

def test_sample(dataset):
    assert len(dataset.images) == 8
    new_dataset1 = sample(dataset, number=8, offset=0)
    new_dataset1_ids = [image.id for image in new_dataset1.images]
    print(new_dataset1_ids)

    new_dataset2 = sample(dataset, number=5, offset=2)
    print([
        image.id for image in new_dataset2.images
    
    ])

    assert len(new_dataset2.images) == 5
    assert new_dataset1_ids[2] == new_dataset2.images[0].id
    assert new_dataset1_ids[6] == new_dataset2.images[4].id

def test_sample_sorted(dataset):
    assert len(dataset.images) == 8
    new_dataset1 = sample(dataset, number=8, offset=0)
    new_dataset1_ids = sorted([image.id for image in new_dataset1.images])
    print(new_dataset1_ids)

    new_dataset2 = sample(dataset, number=5, offset=2, sort="id")
    print([
        image.id for image in new_dataset2.images
    
    ])
    assert len(new_dataset2.images) == 5
    assert new_dataset1_ids[2] == new_dataset2.images[0].id
    assert new_dataset1_ids[6] == new_dataset2.images[4].id

def test_by_ids(dataset):
    filter_ids = [image.id for image in dataset.images[0:2]]
    new_dataset = by_ids(dataset, filter_ids)
    assert len(new_dataset.images) == 2
    assert new_dataset.images[0].id in filter_ids

# Run the test
pytest.main()