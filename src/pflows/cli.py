import os
import json
import argparse
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from typing import Dict, Any
from dotenv import load_dotenv

from pflows.workflow import run_workflow
from compare_models import compare_models, show_model_details
from pflows.viewer.review_images import main as review_main
from train_remote import main as train_main
from run_model import run_and_compare
from compare_dataset_with_standard import compare_datasets


load_dotenv(".env")


def format_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: asdict(value) if is_dataclass(value) else value for key, value in output_data.items()
    }


def run_command(workflow_path: str, output_json: str, env_path: str) -> None:
    output_data: Dict[str, Dict[str, Any]] = {"job": {}}
    output_key = "job"

    try:
        if env_path:
            load_dotenv(env_path)
        output_data = run_workflow(workflow_path, store_dict=output_data, store_dict_key=output_key)
        formatted_output = format_output(output_data)
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                f.write(json.dumps(formatted_output, indent=4))
    except Exception as e:
        raise e


def modal_server_command() -> None:
    current_file = os.path.abspath(__file__)
    main_folder = os.path.abspath(os.path.join(os.path.dirname(current_file), "..", ".."))
    subprocess.run(["modal", "serve", "./src/modal_server.py"], cwd=main_folder)


def train_remote_command(config: str) -> None:
    train_main(config)


def review_command(filepath: str) -> None:
    review_main(filepath)


def main():
    parser = argparse.ArgumentParser(description="pflows command line tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parser for the 'run' command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow_path", type=str, help="Path to the workflow file")
    run_parser.add_argument(
        "--output_json", default=None, type=str, help="Path to the output JSON file"
    )
    run_parser.add_argument("--env", default=None, type=str, help="Path to another env")

    # Parser for the 'modal_server' command
    subparsers.add_parser("modal_server", help="Run modal server")

    # Train server
    train_parser = subparsers.add_parser("train_remote", help="Run a train process")
    train_parser.add_argument("config", type=str, help="Path to the config file")

    # Parser for the 'review' command
    review_parser = subparsers.add_parser("review", help="Run review script")
    review_parser.add_argument("filepath", type=str, help="Path to the file to review")

    # Parser for the 'compare' command
    compare_parser = subparsers.add_parser("compare", help="Compare metrics across multiple models")
    compare_parser.add_argument("folder", help="Root folder containing model metrics")
    compare_parser.add_argument(
        "name_filter", nargs="?", default=None, help="Optional name filter for models"
    )
    compare_parser.add_argument(
        "--format", choices=["markdown", "csv", "tsv"], default="markdown", help="Output format"
    )
    compare_parser.add_argument(
        "--sort", choices=["precision", "recall", "f1"], default=None, help="Sort results by metric"
    )

    # Parser for the 'detail' command
    detail_parser = subparsers.add_parser(
        "detail", help="Show detailed metrics for a specific model"
    )
    detail_parser.add_argument(
        "model_path",
        help="Path to the model file or folder containing the model's metrics.json file",
    )
    detail_parser.add_argument(
        "--format", choices=["markdown", "csv", "tsv"], default="markdown", help="Output format"
    )
    detail_parser.add_argument(
        "--sort",
        choices=["precision", "recall", "f1"],
        default=None,
        help="Sort class results by metric",
    )

    # Parser for the 'run_model' command
    run_model_parser = subparsers.add_parser(
        "run_model", help="Run YOLO model and compare with gold standard"
    )
    run_model_parser.add_argument("model_path", help="Path to YOLO model (.pt file)")
    run_model_parser.add_argument("dataset_path", help="Path to YOLO dataset folder")
    run_model_parser.add_argument("--groups", nargs="+", help="Groups to evaluate (e.g. train val)")
    run_model_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    run_model_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    run_model_parser.add_argument("--output", help="Path to save metrics JSON file")

    # Parser for the 'compare_datasets' command
    compare_datasets_parser = subparsers.add_parser(
        "compare_datasets", help="Compare annotations between two YOLO datasets"
    )
    compare_datasets_parser.add_argument(
        "dataset_path_1", help="Path to first YOLO dataset folder (gold standard)"
    )
    compare_datasets_parser.add_argument(
        "dataset_path_2", help="Path to second YOLO dataset folder"
    )
    compare_datasets_parser.add_argument(
        "--groups", nargs="+", help="Groups to evaluate (e.g. train val)"
    )
    compare_datasets_parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold"
    )
    compare_datasets_parser.add_argument("--output", help="Path to save metrics JSON file")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args.workflow_path, args.output_json, args.env)
    elif args.command == "modal_server":
        modal_server_command()
    elif args.command == "train_remote":
        train_remote_command(args.config)
    elif args.command == "review":
        review_command(args.filepath)
    elif args.command == "compare":
        result = compare_models(args.folder, args.name_filter, args.format, args.sort)
        print(result)
    elif args.command == "detail":
        result = show_model_details(args.model_path, args.format, args.sort)
        print(result)
    elif args.command == "run_model":
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
    elif args.command == "compare_datasets":
        metrics = compare_datasets(
            args.dataset_path_1,
            args.dataset_path_2,
            args.groups,
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
