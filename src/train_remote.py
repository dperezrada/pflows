import sys
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from pflows import workflow
from pflows.tools import pflows_remote


# Load environment variables
def load_environment(override_file: str | None = None) -> Dict[str, Any]:
    """
    Load environment variables from a .env file specified as a command-line argument,
    with an optional override file provided as a parameter.
    Returns a dictionary with the loaded configuration.
    """
    load_dotenv()

    # Load the override file if provided
    if override_file:
        if os.path.exists(override_file):
            load_dotenv(override_file, override=True)
            print(f"Loaded override file: {override_file}")
        else:
            print(f"Warning: Override file {override_file} not found. Skipping.")

    required_vars = [
        "PFLOWS_BASE_URL",
        "PFLOWS_AUTH_TOKEN",
        "YOLO_MODEL",
        "EPOCHS",
        "DATA_FOLDER",
        "OUTPUT_MODEL_NAME",
    ]

    config = {}

    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            raise ValueError(f"{var} must be provided")
        config[var] = value

    # Convert EPOCHS to int
    config["EPOCHS"] = int(config["EPOCHS"])
    config["OUTPUT_MODEL_NAME_PT"] = f"{config['OUTPUT_MODEL_NAME']}.pt"

    return config


# Upload dataset workflow
def create_upload_workflow(config: Dict[str, Any], zip_filepath: str) -> List[Dict[str, Any]]:
    return [
        {
            "task": "base.compress_folder",
            "output": zip_filepath,
            "compress_path": config["DATA_FOLDER"],
        },
        {
            "id": "upload_file",
            "task": "pflows_remote.upload_file",
            "base_url": config["PFLOWS_BASE_URL"],
            "token": config["PFLOWS_AUTH_TOKEN"],
            "local_path": zip_filepath,
            "remote_name": f"{Path(config['DATA_FOLDER']).name}.zip",
        },
    ]


# Remote training workflow
def create_remote_workflow(config: Dict[str, Any], upload_id: str) -> List[Dict[str, Any]]:
    return [
        {
            "task": "base.decompress_zip",
            "zip_path": f"/root/uploads_data/{upload_id}/{Path(config['DATA_FOLDER']).name}.zip",
            "output": "{{REMOTE_TEMP_FOLDER}}/dataset",
        },
        {
            "id": "train",
            "task": "yolo_v8.train",
            "model_name": f"{config['YOLO_MODEL']}.pt",
            "epochs": config["EPOCHS"],
            "data_file": "{{REMOTE_TEMP_FOLDER}}/dataset/data.yaml",
            "model_output": f"{{{{REMOTE_PERSISTED_FOLDER}}}}/models/{config['OUTPUT_MODEL_NAME_PT']}",
        },
    ]


# Process and save metrics
def process_metrics(job_results: Dict[str, Any], metrics_output_folder: str):
    response_json = job_results["train"]["results"]
    metrics = response_json["box"]
    precision, recall, f1_score = metrics["p"], metrics["r"], metrics["f1"]
    ap_class_index = metrics["ap_class_index"]

    class_names = response_json.get("names").values()
    ap_class_names = [response_json["names"][str(i)] for i in ap_class_index]

    aggregated_metrics = {
        "precision": sum(precision) / len(precision),
        "recall": sum(recall) / len(recall),
        "f1_score": sum(f1_score) / len(f1_score),
    }

    confusion_matrix = response_json["confusion_matrix"]["matrix"]

    class_metrics = {}
    for class_index, class_name in enumerate(class_names):
        try:
            ap_index = ap_class_names.index(class_name)
        except ValueError:
            continue
        class_metrics[class_name] = {
            "precision": round(precision[ap_index], 3),
            "recall": round(recall[ap_index], 3),
            "f1_score": round(f1_score[ap_index], 3),
            "n": int(
                sum(
                    [
                        confusion_matrix[row_index][class_index]
                        for row_index in range(len(confusion_matrix))
                    ]
                )
            ),
        }

    sorted_class_metrics = dict(
        sorted(class_metrics.items(), key=lambda x: x[1]["n"], reverse=True)
    )

    metrics_json = {
        "aggregated_metrics": aggregated_metrics,
        "class_metrics": sorted_class_metrics,
    }

    os.makedirs(metrics_output_folder, exist_ok=True)

    with open(f"{metrics_output_folder}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    confusion_matrix_json = {"matrix": confusion_matrix, "class_names": list(class_names)}
    with open(f"{metrics_output_folder}/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrix_json, f, indent=4)

    with open(f"{metrics_output_folder}/results.json", "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=4)

    # Generate confusion matrix plot
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    axis_classes = list(class_names) + ["(background)"]
    plt.xticks(np.arange(len(axis_classes)), axis_classes, rotation=90)
    plt.yticks(np.arange(len(axis_classes)), axis_classes)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{metrics_output_folder}/confusion_matrix.png")

    print(f"Metrics saved to: {metrics_output_folder}")
    print(f"Confusion matrix image: {metrics_output_folder}/confusion_matrix.png")


def main(override_file: str | None = None):
    """
    Main function to run the train server.
    """
    config = {}
    if override_file is None:
        config = load_environment()
    else:
        config = load_environment(override_file)

    zip_filepath = f"dataset_{Path(config['DATA_FOLDER']).name}.zip"

    # Upload dataset
    upload_workflow = create_upload_workflow(config, zip_filepath)
    print(upload_workflow)
    upload_results = workflow.run_workflow(raw_workflow=upload_workflow)
    print(
        f"Upload details: {{'local_path': '{zip_filepath}', 'remote_name': '{Path(config['DATA_FOLDER']).name}.zip'}}"
    )
    print(upload_results)

    print("Waiting for the upload to finish")
    time.sleep(60)

    UPLOAD_ID = upload_results["upload_file"]["upload_id"]
    print(f"/root/uploads_data/{UPLOAD_ID}/{Path(config['DATA_FOLDER']).name}.zip")

    # Remote training
    remote_workflow = create_remote_workflow(config, UPLOAD_ID)
    train_output = pflows_remote.run(
        base_url=config["PFLOWS_BASE_URL"],
        token=config["PFLOWS_AUTH_TOKEN"],
        workflow=remote_workflow,
        env={"UPLOAD_ID": UPLOAD_ID},
        gpu=True,
    )
    job_id = train_output["job_id"]
    print("job_id", job_id)

    while True:
        job_results = pflows_remote.result(
            base_url=config["PFLOWS_BASE_URL"], token=config["PFLOWS_AUTH_TOKEN"], job_id=job_id
        )
        if job_results["status"] == "completed":
            break
        print(job_results)
        time.sleep(30)

    # Process metrics
    metrics_output_folder = f"metrics_{config['OUTPUT_MODEL_NAME']}"
    process_metrics(job_results, metrics_output_folder)

    print("Waiting to download the results")
    time.sleep(180)

    # Retrieve the model
    download_result = pflows_remote.download_job_file(
        base_url=config["PFLOWS_BASE_URL"],
        token=config["PFLOWS_AUTH_TOKEN"],
        job_id=job_id,
        remote_path=f"models/{config['OUTPUT_MODEL_NAME_PT']}",
        local_path=config["OUTPUT_MODEL_NAME_PT"],
    )

    print(download_result)
    print(f"Model saved to: {config['OUTPUT_MODEL_NAME_PT']}")


if __name__ == "__main__":
    main()
