import os
from typing import Tuple
from roboflow import Roboflow
from pflow.utils import log


def parse_url(url: str) -> Tuple[str, str, str]:
    path_url = ""
    if "app.roboflow.com" in url:
        path_url = url.split("app.roboflow.com/")[1]
    else:
        path_url = url.split("universe.roboflow.com/")[1]
    user = path_url.split("/")[0]
    project = path_url.split("/")[1]
    dataset_version = path_url.split("/")[-1]
    return user, project, dataset_version


def download_dataset(url: str, target_dir: str) -> bool:
    if os.path.exists(target_dir):
        log("roboflow_tools", "download_dataset", "Already downloaded dataset")
        return False

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    user, project_name, dataset_version = parse_url(url)

    log("roboflow_tools", "download_dataset", "Downloading dataset")
    project = rf.workspace(user).project(project_name)
    os.makedirs(target_dir, exist_ok=True)
    project.version(dataset_version).download("yolov8", target_dir)
    return True
