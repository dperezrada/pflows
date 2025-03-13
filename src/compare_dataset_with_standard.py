from pathlib import Path
from typing import Dict, Any, List
import json
import tempfile
from pflows.tools.yolo_v8 import load_dataset, run_model
from pflows.tools.compare_datasets import compare_images_annotations
from pflows.tools.filter_images import by_group, by_groups
from pflows.typedef import Dataset, Image
from pflows.tools.external import run_function_path


def compare_datasets(
    dataset_path_1: str,
    dataset_path_2: str,
    groups: str | List[str] | None = None,
    iou_threshold: float = 0.5,
    output_metrics: str | None = None,
) -> Dict[str, Any]:
    """
    Compare two YOLO datasets.

    Args:
        dataset_path_1: Path to the first YOLO dataset folder (treated as gold standard)
        dataset_path_2: Path to the second YOLO dataset folder
        groups: Single group name or list of group names to filter by (e.g. "train", ["train", "val"])
        iou_threshold: IoU threshold for matching annotations
        output_metrics: Optional path to save metrics JSON file

    Returns:
        Dictionary containing comparison metrics
    """
    # Load both datasets
    dataset_1 = load_dataset(None, dataset_path_1)
    dataset_2 = load_dataset(None, dataset_path_2)

    # Filter by groups if specified
    if groups is not None:
        if isinstance(groups, str):
            dataset_1 = by_group(dataset_1, groups)
            dataset_2 = by_group(dataset_2, groups)
        else:
            dataset_1 = by_groups(dataset_1, groups)
            dataset_2 = by_groups(dataset_2, groups)

    # Get category names from first dataset
    category_names = [category.name for category in dataset_1.categories]

    # Compare results
    metrics = compare_images_annotations(
        dataset_1.images, dataset_2.images, category_names, iou_threshold=iou_threshold
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

    parser = argparse.ArgumentParser(description="Compare two YOLO datasets")
    parser.add_argument("dataset_path_1", help="Path to first YOLO dataset folder (gold standard)")
    parser.add_argument("dataset_path_2", help="Path to second YOLO dataset folder")
    parser.add_argument("--groups", nargs="+", help="Groups to evaluate (e.g. train val)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output", help="Path to save metrics JSON file")

    args = parser.parse_args()

    metrics = compare_datasets(
        args.dataset_path_1,
        args.dataset_path_2,
        args.groups,
        args.iou_threshold,
        args.output,
    )

    # Print summary metrics
    print("\nOverall Metrics:")
    print(
        f"Total Annotations: {metrics['overall']['TP'] + metrics['overall']['FP'] + metrics['overall']['FN']}"
    )
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {metrics['overall']['f1_score']:.4f}")
    print(f"True Positives: {metrics['overall']['TP']}")
    print(f"False Positives: {metrics['overall']['FP']}")
    print(f"False Negatives: {metrics['overall']['FN']}")

    print("\nMetrics by Category:")
    for category, cat_metrics in metrics["categories"].items():
        print(f"\n{category}:")
        print(f"  Precision: {cat_metrics['precision']:.4f}")
        print(f"  Recall: {cat_metrics['recall']:.4f}")
        print(f"  F1 Score: {cat_metrics['f1_score']:.4f}")
        print(f"  True Positives: {cat_metrics['TP']}")
        print(f"  False Positives: {cat_metrics['FP']}")
        print(f"  False Negatives: {cat_metrics['FN']}")
        print(f"  Total Annotations: {cat_metrics['TP'] + cat_metrics['FP'] + cat_metrics['FN']}")

    # Create a temporary file and store the metrics in tsv format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv", mode="w", encoding="utf-8") as f:
        tsv_path = f.name
        print(f"Metrics saved to temporary file: {tsv_path}")
        # write the header
        f.write("category\ttotal_annotations\tprecision\trecall\tf1_score\tTP\tFP\tFN\n")
        total_annotations = (
            metrics["overall"]["TP"] + metrics["overall"]["FP"] + metrics["overall"]["FN"]
        )
        f.write(
            f"Overall\t{total_annotations}\t{metrics['overall']['precision']:.4f}\t"
            f"{metrics['overall']['recall']:.4f}\t{metrics['overall']['f1_score']:.4f}\t"
            f"{metrics['overall']['TP']}\t{metrics['overall']['FP']}\t{metrics['overall']['FN']}\n"
        )
        for category, cat_metrics in metrics["categories"].items():
            category_total = cat_metrics["TP"] + cat_metrics["FP"] + cat_metrics["FN"]
            f.write(
                f"{category}\t{category_total}\t{cat_metrics['precision']:.4f}\t"
                f"{cat_metrics['recall']:.4f}\t{cat_metrics['f1_score']:.4f}\t{cat_metrics['TP']}\t"
                f"{cat_metrics['FP']}\t{cat_metrics['FN']}\n"
            )
