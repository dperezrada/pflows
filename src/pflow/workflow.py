import os
import re
import json
import inspect

from typing import Tuple, Dict, Any, Callable, List
from pflow.typedef import Dataset, Task


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
    try:
        task_file, task_function_name = task_name.split(".")
        task_module = __import__(f"pflow.tools.{task_file}", fromlist=[task_function_name])
        task_function = getattr(task_module, task_function_name)
        params = load_function_args(task_function)
    except Exception as exc:
        print(f"Error loading task: {task_name}", exc)
        raise ValueError(f"The task '{task_name}' is not a valid task.") from exc

    return task_function, params


def replace_variables(text: str, current_dir: str) -> str:
    # we are going to search for {{variable}} and replace it with the value of the variable
    # we are going to use a regular expression to find all the variables
    matches = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", text)
    for match in matches:
        value = os.getenv(match)
        if value is None or value == "":
            if match == "CURRENT_DIR":
                value = current_dir
            else:
                raise ValueError(f"The variable '{match}' is not defined.")
        text = text.replace(f"{{{{{match}}}}}", value)
    return text


# pylint: disable=too-many-locals,too-many-branches
def read_workflow(workflow_path: str) -> Tuple[List[Task], Dict[str, Any]]:
    # Load the workflow and check is a valid JSON
    if not os.path.exists(workflow_path):
        raise FileNotFoundError("The workflow file does not exist.")
    workflow_dir = os.path.abspath(os.path.dirname(workflow_path))

    with open(workflow_path, "r", encoding="utf-8") as f:
        try:
            workflow_text = f.read()
            workflow = json.loads(workflow_text)
        except json.JSONDecodeError as exc:
            raise ValueError("The workflow file is not a valid JSON file.") from exc

    non_set_env_tasks_found = False
    workflow_tasks = []
    for index, task in enumerate(workflow):
        if task["task"] == "set_env_var":
            if non_set_env_tasks_found:
                raise ValueError("set_env_var tasks must be the first tasks in the workflow.")
            os.environ[task["name"]] = task["value"]
            continue
        workflow_tasks.append(task)
        non_set_env_tasks_found = True

    workflow_text = json.dumps(workflow_tasks)
    workflow_text = replace_variables(workflow_text, workflow_dir)
    workflow = json.loads(workflow_text)

    workflow_basepath = os.path.abspath(os.path.dirname(workflow_path))

    workflow_data = {"dataset": Dataset(images=[], categories=[], groups=[])}
    workflow_reviewed_tasks: List[Task] = []
    for index, raw_task in enumerate(workflow):
        if "task" not in raw_task:
            raise ValueError(f"The 'task' key is missing in one of the task {index +1}.")
        task_name = raw_task["task"]
        task_args = raw_task.copy()
        del task_args["task"]

        task_function, params = load_function(task_name)
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
                raise ValueError(
                    f"The parameter '{param}' is required for task {index +1}: {raw_task['task']}."
                )
        # check if there are any extra parameters
        for param in task_args:
            if param not in params:
                if param in workflow_data:
                    task_args[param] = workflow_data[param]
                    continue
                raise ValueError(f"The parameter '{param}' is not valid for task {index +1}.")
        task = Task(task=task_name, function=task_function, params=task_args)
        workflow_reviewed_tasks.append(task)

    return workflow_reviewed_tasks, workflow_data


def run_workflow(workflow_path: str) -> Dict[str, Any]:
    # Load the workflow and check is a valid JSON
    workflow, workflow_data = read_workflow(workflow_path)
    for task in workflow:
        print("")
        print("-" * 20, task.task, "-" * 20)
        params = {
            param: (
                task.params[param]
                if task.params[param] != "__workflow_parameter__"
                else workflow_data[param]
            )
            for param in task.params
        }
        result = task.function(**params)
        if result is not None and isinstance(result, dict):
            workflow_data.update(result)
        if result is not None and isinstance(result, Dataset):
            workflow_data["dataset"] = result
    return workflow_data
