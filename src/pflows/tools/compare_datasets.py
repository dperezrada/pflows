from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from pflows.typedef import Annotation, Image


def calculate_iou(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> float:
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


def find_best_match(
    gold_annotation: Annotation, new_annotations: List[Annotation]
) -> Tuple[Annotation | None, float]:
    best_match = None
    best_iou = 0.0

    for new_annotation in new_annotations:
        if (
            gold_annotation.category_name == new_annotation.category_name
            and gold_annotation.bbox is not None
            and new_annotation.bbox is not None
        ):
            iou = calculate_iou(gold_annotation.bbox, new_annotation.bbox)
            if iou > best_iou:
                best_match = new_annotation
                best_iou = iou

    return (best_match, best_iou)


def compare_annotations(
    gold_annotation: Annotation, infer_annotations: List[Annotation], iou_threshold: float = 0.5
) -> Dict[str, Any]:
    category_name = gold_annotation.category_name

    # Find the best matching annotation in the new image
    best_match, best_iou = find_best_match(gold_annotation, infer_annotations)

    status = "not found"
    confidence = 0.0
    if best_match is not None:
        if best_iou >= iou_threshold:
            status = "found"
            confidence = best_match.conf

    return {
        "id": gold_annotation.id,
        "matched_id": best_match.id if best_match else "",
        "status": status,
        "confidence": confidence,
        "iou": best_iou,
    }


def calculate_metrics(
    original_annotations: List[Annotation],
    inferred_annotations: List[Annotation],
    category_names: Sequence[str],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    matched_inferred = set()
    incorrect_annotations = {"FP": [], "FN": []}

    for gold_annotation in original_annotations:
        result = compare_annotations(gold_annotation, inferred_annotations, iou_threshold)
        category = gold_annotation.category_name
        if category not in category_names:
            continue
        if result["status"] == "found":
            metrics[category]["TP"] += 1
            matched_inferred.add(result["matched_id"])
        else:
            metrics[category]["FN"] += 1
            incorrect_annotations["FN"].append(gold_annotation)

    for inferred_annotation in inferred_annotations:
        if inferred_annotation.category_name not in category_names:
            continue
        category = inferred_annotation.category_name
        if inferred_annotation.id not in matched_inferred:
            metrics[category]["FP"] += 1
            incorrect_annotations["FP"].append(inferred_annotation)

    return dict(metrics), incorrect_annotations


def compare_images_annotations(
    gold_images: Sequence[Image],
    inferred_images: Sequence[Image],
    category_names: Sequence[str],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    gold_ids = set(image.id for image in gold_images)
    infer_ids = set(image.id for image in inferred_images)
    infer_ids = infer_ids.union(
        set(image_id for image in inferred_images for image_id in image.intermediate_ids)
    )
    common_ids = gold_ids.intersection(infer_ids)

    overall_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    incorrect_images = {}

    for image_id in common_ids:
        original_image = next(img for img in gold_images if img.id == image_id)
        inferred_image = next(img for img in inferred_images if img.id == image_id)
        image_metrics, incorrect_annotations = calculate_metrics(
            original_image.annotations, inferred_image.annotations, category_names, iou_threshold
        )

        if incorrect_annotations["FP"] or incorrect_annotations["FN"]:
            incorrect_images[image_id] = incorrect_annotations

        for category, metrics in image_metrics.items():
            overall_metrics[category]["TP"] += metrics["TP"]
            overall_metrics[category]["FP"] += metrics["FP"]
            overall_metrics[category]["FN"] += metrics["FN"]

        # Build confusion matrix
        for gold_ann in original_image.annotations:
            best_match = None
            best_iou = 0
            if gold_ann.category_name not in category_names:
                continue
            for infer_ann in inferred_image.annotations:
                if infer_ann.category_name not in category_names:
                    continue
                if not gold_ann.bbox or not infer_ann.bbox:
                    continue
                iou = calculate_iou(gold_ann.bbox, infer_ann.bbox)
                if iou > best_iou:
                    best_match = infer_ann
                    best_iou = iou

            if best_match and best_iou >= iou_threshold:
                confusion_matrix[gold_ann.category_name][best_match.category_name] += 1
            else:
                confusion_matrix[gold_ann.category_name]["not_detected"] += 1

    category_metrics = {}
    for category, metrics in overall_metrics.items():
        tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

    total_tp = sum(metrics["TP"] for metrics in overall_metrics.values())
    total_fp = sum(metrics["FP"] for metrics in overall_metrics.values())
    total_fn = sum(metrics["FN"] for metrics in overall_metrics.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1_score = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    return {
        "raw": overall_metrics,
        "categories": category_metrics,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1_score,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
        },
        "confusion_matrix": confusion_matrix,
        "incorrect_images": incorrect_images,
    }
