import os
import json
from typing import Dict, List, Optional, Any
from io import StringIO
import csv
from tabulate import tabulate


def find_metrics_files(root_folder: str, name_filter: Optional[str] = None) -> List[str]:
    metrics_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file == "metrics.json":
                if name_filter is None or name_filter in root:
                    metrics_files.append(os.path.join(root, file))
    return metrics_files


def read_metrics_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        to_return: Dict[str, Any] = json.load(f)
        return to_return


def extract_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    aggregated = data["aggregated_metrics"]
    return {
        "precision": aggregated["precision"],
        "recall": aggregated["recall"],
        "f1_score": aggregated["f1_score"],
    }


def create_comparison_table(
    metrics_data: Dict[str, Dict[str, float]],
    output_format: str = "markdown",
    sort_by: Optional[str] = None,
) -> str:
    headers = ["Folder", "Precision", "Recall", "F1 Score"]
    table_data = [
        [folder, metrics["precision"], metrics["recall"], metrics["f1_score"]]
        for folder, metrics in metrics_data.items()
    ]

    if sort_by:
        sort_index = ["precision", "recall", "f1_score"].index(sort_by) + 1
        table_data.sort(key=lambda x: x[sort_index], reverse=True)

    if output_format == "markdown":
        return tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=".4f")
    if output_format in ["csv", "tsv"]:
        delimiter = "," if output_format == "csv" else "\t"
        output = StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(table_data)
        return output.getvalue()
    raise ValueError(f"Unsupported format: {output_format}")


def show_detailed_metrics(
    data: Dict[str, Any],
    model_name: str,
    output_format: str = "markdown",
    sort_by: Optional[str] = None,
) -> str:
    aggregated = data["aggregated_metrics"]
    class_metrics = data["class_metrics"]

    agg_data = [
        ["Precision", aggregated["precision"]],
        ["Recall", aggregated["recall"]],
        ["F1 Score", aggregated["f1_score"]],
    ]

    class_data = [
        [class_name, metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["n"]]
        for class_name, metrics in class_metrics.items()
    ]

    if sort_by:
        sort_index = ["precision", "recall", "f1_score"].index(sort_by) + 1
        class_data.sort(key=lambda x: x[sort_index], reverse=True)

    if output_format == "markdown":
        agg_table = tabulate(agg_data, headers=["Metric", "Value"], tablefmt="pipe", floatfmt=".4f")
        class_table = tabulate(
            class_data,
            headers=["Class", "Precision", "Recall", "F1 Score", "Count"],
            tablefmt="pipe",
            floatfmt=".4f",
        )
        return (
            f"Detailed Metrics for {model_name}\n\n"
            f"Aggregated Metrics:\n{agg_table}\n\n"
            f"Class-specific Metrics:\n{class_table}"
        )
    if output_format in ["csv", "tsv"]:
        delimiter = "," if output_format == "csv" else "\t"
        output = StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        writer.writerow(["Metric", "Value"])
        writer.writerows(agg_data)
        writer.writerow([])  # Empty row as separator
        writer.writerow(["Class", "Precision", "Recall", "F1 Score", "Count"])
        writer.writerows(class_data)
        return output.getvalue()
    raise ValueError(f"Unsupported format: {output_format}")


def compare_models(
    root_folder: str,
    name_filter: Optional[str] = None,
    output_format: str = "markdown",
    sort_by: Optional[str] = None,
) -> str:
    metrics_files = find_metrics_files(root_folder, name_filter)
    metrics_data: Dict[str, Dict[str, float]] = {}

    for file_path in metrics_files:
        folder = os.path.dirname(file_path)
        data = read_metrics_file(file_path)
        metrics = extract_metrics(data)
        metrics_data[folder] = metrics

    return create_comparison_table(metrics_data, output_format, sort_by)


def find_metrics_file(model_path: str) -> Optional[str]:
    if os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)
    else:
        model_dir = model_path

    for root, _, files in os.walk(model_dir):
        if "metrics.json" in files:
            return os.path.join(root, "metrics.json")

    return None


def show_model_details(
    model_path: str, output_format: str = "markdown", sort_by: Optional[str] = None
) -> str:
    metrics_file = find_metrics_file(model_path)

    if not metrics_file:
        return f"Error: metrics.json not found for model {model_path}"

    data = read_metrics_file(metrics_file)
    return show_detailed_metrics(data, model_path, output_format, sort_by)
