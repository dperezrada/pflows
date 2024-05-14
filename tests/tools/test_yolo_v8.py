import pytest
from pathlib import Path
from pflow.tools.yolo_v8 import run_model
from pflow.tools.filter_images import sample
from pflow.typedef import Dataset
from pflow.tools.annotations import filter_by_tag

model_path = Path(__file__).parent / "yolov8n.pt"

def test_run_model(dataset):
    sampled_dataset = sample(dataset, number=1, offset=0).get("dataset")
    total_initial_annotations = len(sampled_dataset.images[0].annotations)
    new_dataset = run_model(sampled_dataset, model_path, add_tag="yolov8n").get("dataset")
    image_data = new_dataset.images[0]
    assert len(image_data.annotations) > total_initial_annotations
    filter_annotations = filter_by_tag(image_data.annotations, "yolov8n")
    assert len(filter_annotations) > 0
    assert filter_annotations[0].category_name == 'bird'

# Run the test
pytest.main()