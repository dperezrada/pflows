from typing import Dict, List, Tuple

from pflows.typedef import Annotation

def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    # Calculate the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou

def find_best_match(gold_annotation: Annotation, new_annotations: List[Annotation]) -> Tuple[Annotation, float]:
    best_match = None
    best_iou = 0.0

    for new_annotation in new_annotations:
        if gold_annotation.category_name == new_annotation.category_name and gold_annotation.bbox is not None and new_annotation.bbox is not None:
            iou = calculate_iou(gold_annotation.bbox, new_annotation.bbox)
            if iou > best_iou:
                best_match = new_annotation
                best_iou = iou

    return best_match, best_iou

def compare_annotations(gold_annotation: Annotation, new_dataset: Dataset, iou_threshold: float = 0.5) -> Dict[str, Any]:
    image_id = gold_annotation.image_id
    category_name = gold_annotation.category_name

    # Find the corresponding image in the new dataset
    new_image = next((image for image in new_dataset.images if image.id == image_id), None)

    if new_image is None:
        return {
            "image_id": image_id,
            "category_name": category_name,
            "status": "FN",
            "iou": 0.0,
            "confidence": 0.0
        }

    # Find the best matching annotation in the new image
    best_match, best_iou = find_best_match(gold_annotation, new_image.annotations)

    if best_match is None or best_iou < iou_threshold:
        return {
            "image_id": image_id,
            "category_name": category_name,
            "status": "FN",
            "iou": best_iou,
            "confidence": 0.0
        }

    return {
        "image_id": image_id,
        "category_name": category_name,
        "status": "TP" if best_iou >= iou_threshold else "FP",
        "iou": best_iou,
        "confidence": best_match.conf
    }