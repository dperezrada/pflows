from pathlib import Path
from typing import Dict, Any, List
import json

from pflows.tools.yolo_v8 import load_dataset, run_model
from pflows.tools.compare_datasets import compare_images_annotations
from pflows.tools.filter_images import by_group, by_groups
from pflows.typedef import Dataset, Image


def run_and_compare(
    model_path: str,
    dataset_path: str,
    groups: str | List[str] | None = None,
    threshold: float = 0.5,
    iou_threshold: float = 0.5,
    output_metrics: str | None = None,
) -> Dict[str, Any]:
    """
    Run YOLO model on images and compare with gold standard annotations.

    Args:
        model_path: Path to the YOLO model (.pt file)
        dataset_path: Path to the YOLO dataset folder
        groups: Single group name or list of group names to filter by (e.g. "train", ["train", "val"])
        threshold: Confidence threshold for model predictions
        iou_threshold: IoU threshold for matching predictions with ground truth
        output_metrics: Optional path to save metrics JSON file

    Returns:
        Dictionary containing comparison metrics
    """
    # Load gold standard dataset
    gold_dataset = load_dataset(None, dataset_path)

    # Filter by groups if specified
    if groups is not None:
        if isinstance(groups, str):
            gold_dataset = by_group(gold_dataset, groups)
        else:
            gold_dataset = by_groups(gold_dataset, groups)

    new_dataset_to_infer = Dataset(
        images=[
            Image(**{**image.copy().__dict__, "annotations": []}) for image in gold_dataset.images
        ],
        categories=gold_dataset.categories,
        groups=gold_dataset.groups,
    )
    # Run model on images
    inferred_dataset = run_model(
        dataset=new_dataset_to_infer,
        model_path=model_path,
        threshold=threshold,
        add_tag="inference",
    )
    # keep only

    # Get category names from gold dataset
    category_names = [category.name for category in gold_dataset.categories]

    # Compare results
    metrics = compare_images_annotations(
        gold_dataset.images, inferred_dataset.images, category_names, iou_threshold=iou_threshold
    )

    if output_metrics:
        # Save metrics to file
        metrics_dir = Path(output_metrics).parent
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(output_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to: {output_metrics}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO model and compare with gold standard")
    parser.add_argument("model_path", help="Path to YOLO model (.pt file)")
    parser.add_argument("dataset_path", help="Path to YOLO dataset folder")
    parser.add_argument("--groups", nargs="+", help="Groups to evaluate (e.g. train val)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output", help="Path to save metrics JSON file")

    args = parser.parse_args()

    metrics = run_and_compare(
        args.model_path,
        args.dataset_path,
        args.groups,
        args.threshold,
        args.iou_threshold,
        args.output,
    )

    # Print summary metrics
    print("\nOverall Metrics:")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {metrics['overall']['f1_score']:.4f}")

    print("\nMetrics by Category:")
    for category, cat_metrics in metrics["categories"].items():
        print(f"\n{category}:")
        print(f"  Precision: {cat_metrics['precision']:.4f}")
        print(f"  Recall: {cat_metrics['recall']:.4f}")
        print(f"  F1 Score: {cat_metrics['f1_score']:.4f}")
