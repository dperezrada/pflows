"""Console script for pflows."""

import json
import argparse
from dataclasses import asdict, is_dataclass
from typing import Dict, Any
from dotenv import load_dotenv

from pflows.workflow import run_workflow

load_dotenv()


def format_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: asdict(value) if is_dataclass(value) else value for key, value in output_data.items()
    }


def main() -> None:
    """Run a workflow."""
    parser = argparse.ArgumentParser(description="Run a workflow.")
    parser.add_argument("workflow_path", type=str, help="Path to the workflow file")
    parser.add_argument(
        "--output_json", default=None, type=str, help="Path to the output JSON file"
    )
    args = parser.parse_args()

    worflow_path = args.workflow_path
    output_json = args.output_json

    if not worflow_path:
        parser.error("The 'workflow_path' argument is required.")
    output_data: Dict[str, Dict[str, Any]] = {"job": {}}
    output_key = "job"

    try:
        output_data = run_workflow(
            args.workflow_path, store_dict=output_data, store_dict_key=output_key
        )
        formatted_output = format_output(output_data)
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                f.write(json.dumps(formatted_output, indent=4))
    except Exception as e:
        raise e
        # print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
