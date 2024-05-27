import pytest
from pathlib import Path
from pflows.tools.yolo_v8 import process_image_annotations, run_model
from pflows.tools.filter_images import sample
from pflows.typedef import Dataset, Category
from pflows.tools.annotations import filter_by_tag

model_path = Path(__file__).parent / "yolov8n.pt"


def test_run_model(dataset):
    sampled_dataset = sample(dataset, number=1, offset=6, sort="id")
    total_initial_annotations = len(sampled_dataset.images[0].annotations)
    new_dataset = run_model(sampled_dataset, str(model_path), add_tag="yolov8n")
    image_data = new_dataset.images[0]
    assert len(image_data.annotations) > total_initial_annotations
    filter_annotations = filter_by_tag(image_data.annotations, "yolov8n")
    assert len(filter_annotations) > 0
    assert filter_annotations[0].category_name == "bird"


def test_process_image_annotations():
    current_dir = Path(__file__).parent
    annotation_path = (
        current_dir
        / "../fixtures/Basketball Players.v1-original_raw-images.yolov8/train/labels/youtube-5_jpg.rf.4daa9c8824fdabdfee66c62adae476e1.txt"
    )
    annotations = process_image_annotations(
        str(annotation_path),
        [
            Category(name=name, id=i)
            for i, name in enumerate(
                [
                "Ball",
                "Hoop",
                "Period",
                "Player",
                "Ref",
                "Shot Clock",
                "Team Name",
                "Team Points",
                "Time Remaining",
            ])
        ]
    )
    assert 22 == len(annotations)
    assert annotations[0].category_name == "Ref"
    assert annotations[0].bbox == (0.302864, 0.688493, 0.355481, 0.794063)


# Run the test
pytest.main()
