import os
import re
import json
import inspect

from typing import Tuple, Dict, Any, Callable, TypedDict, List
from pflow.typedef import Dataset


class TaskObj(TypedDict):
    task: str
    _function: Callable[..., Any]
    _params: Dict[str, Any]


def load_function_args(function: Callable[..., Any]) -> Dict[str, Dict[str, str | bool]]:
    # Get the function signature
    signature = inspect.signature(function)
    # Create a dictionary to store the parameter information
    param_info = {}

    # Iterate over the parameters and populate the dictionary
    for param in signature.parameters.values():
        param_info[param.name] = {
            "type": param.annotation,
            "required": param.default == inspect.Parameter.empty,
        }
    return param_info


def load_function(task_name: str) -> Tuple[Any, Dict[str, Dict[str, str | bool]]]:
    task_file, task_function_name = task_name.split(".")
    task_module = __import__(f"pflow.tools.{task_file}", fromlist=[task_function_name])
    task_function = getattr(task_module, task_function_name)
    params = load_function_args(task_function)
    return task_function, params


# pylint: disable=too-many-locals
def read_workflow(workflow_path: str) -> Tuple[List[TaskObj], Dict[str, Any]]:
    # Load the workflow and check is a valid JSON
    if not os.path.exists(workflow_path):
        raise FileNotFoundError("The workflow file does not exist.")

    with open(workflow_path, "r", encoding="utf-8") as f:
        try:
            workflow = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError("The workflow file is not a valid JSON file.") from exc

    workflow_basepath = os.path.abspath(os.path.dirname(workflow_path))

    workflow_data = {"dataset": Dataset(images=[], categories=[])}
    for index, task in enumerate(workflow):
        if "task" not in task:
            raise ValueError(f"The 'task' key is missing in one of the task {index +1}.")
        task_name = task["task"]
        task_args = task.copy()
        del task_args["task"]

        task_function, params = load_function(task_name)
        task["_function"] = task_function
        # check if all required parameters are present
        for param, info in params.items():
            if info["required"] and param not in task_args:
                possible_relatives = [
                    task_param
                    for task_param in task_args
                    if task_param.endswith("_relative")
                    and re.sub(r"_relative$", "", task_param) == param
                ]
                if len(possible_relatives) > 0:
                    relative_param = possible_relatives[0]
                    task_args[param] = os.path.abspath(
                        os.path.join(workflow_basepath, task_args[relative_param])
                    )
                    del task_args[relative_param]
                    continue
                if param in workflow_data:
                    task_args[param] = "__workflow_parameter__"
                    continue
                raise ValueError(f"The parameter '{param}' is required for task {index +1}.")
        # check if there are any extra parameters
        for param in task_args:
            if param not in params:
                if param in workflow_data:
                    task_args[param] = workflow_data[param]
                    continue
                raise ValueError(f"The parameter '{param}' is not valid for task {index +1}.")
        task["_params"] = task_args
    return workflow, workflow_data


def run_workflow(workflow_path: str) -> Dict[str, Any]:
    # Load the workflow and check is a valid JSON
    workflow, workflow_data = read_workflow(workflow_path)
    for task in workflow:
        print("running task", task["task"])
        params = {
            param: (
                task["_params"][param]
                if task["_params"][param] != "__workflow_parameter__"
                else workflow_data[param]
            )
            for param in task["_params"]
        }
        result = task["_function"](**params)
        if result is not None and isinstance(result, dict):
            workflow_data.update(result)
    return workflow_data
