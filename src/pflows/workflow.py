import os
import re
import json
import inspect

from typing import Sequence, Tuple, Dict, Any, Callable, List, cast
from pflows.typedef import Dataset, Task


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
        task_module = __import__(f"pflows.tools.{task_file}", fromlist=[task_function_name])
        task_function = getattr(task_module, task_function_name)
        params = load_function_args(task_function)
    except Exception as exc:
        print(f"Error loading task: {task_name}", exc)
        raise ValueError(f"The task '{task_name}' is not a valid task.") from exc

    return task_function, params


def replace_variables(text: str, current_dir: str | None = None) -> str:
    # we are going to search for {{variable}} and replace it with the value of the variable
    # we are going to use a regular expression to find all the variables
    matches = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", text)
    for match in matches:
        value = os.getenv(match)
        if value is None or value == "":
            if match == "CURRENT_DIR" and current_dir is not None:
                value = current_dir
            else:
                found_base_folder = os.getenv("BASE_FOLDER")
                if match == "PERSISTED_FOLDER" and found_base_folder is not None:
                    value = found_base_folder
                else:
                    print(f"Variable {match} not found in environment variables.")
                    continue
        text = text.replace(f"{{{{{match}}}}}", value)
    return text


def prepare_workflow(
    raw_workflow: Sequence[Dict[str, Any]], workflow_dir: str | None
) -> Sequence[Dict[str, Any]]:
    non_set_env_tasks_found = False
    workflow_tasks = []
    for task in raw_workflow:
        if task["task"] == "set_env_var":
            if non_set_env_tasks_found:
                raise ValueError("set_env_var tasks must be the first tasks in the workflow.")
            os.environ[task["name"]] = task["value"]
            continue
        workflow_tasks.append(task)
        non_set_env_tasks_found = True

    # Keep in a variable all the functions that are going to be used in the workflow
    workflow_functions = {}
    for index, task in enumerate(workflow_tasks):
        if task.get("function"):
            workflow_functions[f"task_{index}"] = task["function"]
            task["function"] = "run_function_path"

    # Deal with the model_path option possible as a path or as a YOLO object
    yolo_model_objects = {}
    for index, task in enumerate(workflow_tasks):
        print(task)
        if task.get("model_path") and not isinstance(task["model_path"], str):
            yolo_model_objects[f"task_{index}"] = task["model_path"]
            task["model_path"] = "run_yolo_model_path"

    workflow_text = json.dumps(workflow_tasks)
    workflow_text = replace_variables(workflow_text, workflow_dir)
    workflow = cast(Sequence[Dict[str, Any]], json.loads(workflow_text))
    for index, task in enumerate(workflow):
        if task.get("function"):
            task["function"] = workflow_functions[f"task_{index}"]
        if task.get("model_path") and task["model_path"] == "run_yolo_model_path":
            task["model_path"] = yolo_model_objects[f"task_{index}"]
    return workflow


# pylint: disable=too-many-locals,too-many-branches
def read_workflow(
    workflow_path: str | None = None, raw_workflow: Sequence[Dict[str, Any]] | None = None
) -> Tuple[Sequence[Task], Dict[str, Any]]:
    if workflow_path is None and raw_workflow is None:
        raise ValueError("You must provide a workflow path or a workflow.")
    workflow_dir = None
    if workflow_path is not None:
        raw_workflow, workflow_dir = read_local_workflow(workflow_path)
    if raw_workflow is None:
        raise ValueError("You must provide a workflow path or a workflow.")

    workflow = prepare_workflow(raw_workflow, workflow_dir)

    workflow_data = {"dataset": Dataset(images=[], categories=[], groups=[])}
    workflow_reviewed_tasks: List[Task] = []
    for index, raw_task in enumerate(workflow):
        if "task" not in raw_task:
            raise ValueError(f"The 'task' key is missing in one of the task {index +1}.")
        task_name = raw_task["task"]
        task_args = raw_task.copy()
        del task_args["task"]

        task_function, params = load_function(task_name)
        skip_task = False
        task_id = None
        # check if all required parameters are present
        for param, info in params.items():
            if info["required"] and param not in task_args:
                possible_relatives = [
                    task_param
                    for task_param in task_args
                    if task_param.endswith("_relative")
                    and re.sub(r"_relative$", "", task_param) == param
                ]
                if len(possible_relatives) > 0 and workflow_dir is not None:
                    relative_param = possible_relatives[0]
                    task_args[param] = os.path.abspath(
                        os.path.join(workflow_dir, task_args[relative_param])
                    )
                    del task_args[relative_param]
                    continue
                if param in workflow_data:
                    task_args[param] = "__workflow_parameter__"
                    continue
                raise ValueError(
                    f"The parameter '{param}' is required for task {index +1}: {raw_task['task']}."
                )
        if "skip" in task_args:
            skip_task = True
            del task_args["skip"]
        if "id" in task_args:
            task_id = task_args["id"]
            del task_args["id"]

        # check if there are any extra parameters
        for param in task_args:
            if param not in params:
                if param in workflow_data:
                    task_args[param] = workflow_data[param]
                    continue
                raise ValueError(f"The parameter '{param}' is not valid for task {index +1}.")
        task = Task(
            task=task_name, function=task_function, params=task_args, skip=skip_task, id=task_id
        )
        workflow_reviewed_tasks.append(task)

    return workflow_reviewed_tasks, workflow_data


def read_local_workflow(workflow_path: str) -> Tuple[Sequence[Dict[str, Any]], str]:
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
    return workflow, workflow_dir


def internal_run_workflow(
    workflow: Sequence[Task],
    workflow_data: Dict[str, Any],
    store_dict: Dict[str, Any] | None = None,
    store_dict_key: str | None = None,
    id_output_data: Dict[str, Any] = {},
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for task in workflow:
        print("")
        print("-" * 20, task.task, "-" * 20)
        if task.skip:
            print("Skipping task.")
            continue
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
            # get result without the dataset key
            if task.id is not None:
                id_output_data[task.id] = {
                    key: value for key, value in result.items() if key != "dataset"
                }
                if store_dict is not None and store_dict_key is not None and result != {}:
                    store_dict[store_dict_key] = {
                        **store_dict[store_dict_key],
                        **id_output_data,
                    }
        if result is not None and isinstance(result, Dataset):
            workflow_data["dataset"] = result
    return workflow_data, id_output_data


def run_workflow(
    workflow_path: str | None = None,
    raw_workflow: Sequence[Dict[str, Any]] | None = None,
    store_dict: Dict[str, Any] | None = None,
    store_dict_key: str | None = None,
) -> Dict[str, Any]:
    workflow, workflow_data = read_workflow(workflow_path, raw_workflow)
    id_output_data = {}
    try:
        workflow_data, id_output_data = internal_run_workflow(
            workflow, workflow_data, store_dict, store_dict_key, id_output_data
        )

    except SystemExit:
        pass
    if store_dict is not None and store_dict_key is not None:
        store_dict[store_dict_key] = {**store_dict[store_dict_key], **id_output_data}
        return cast(Dict[str, Any], store_dict[store_dict_key])
    return id_output_data


def run_workflow_dataset(
    workflow_path: str | None = None,
    raw_workflow: Sequence[Dict[str, Any]] | None = None,
    store_dict: Dict[str, Any] | None = None,
    store_dict_key: str | None = None,
    dataset: Dataset | None = None,
) -> Dataset:
    workflow, workflow_data = read_workflow(workflow_path, raw_workflow)
    if dataset is not None:
        workflow_data["dataset"] = dataset
    try:
        workflow_data, _ = internal_run_workflow(
            workflow, workflow_data, store_dict, store_dict_key
        )
    except SystemExit:
        pass
    return Dataset.from_dict(workflow_data["dataset"])
