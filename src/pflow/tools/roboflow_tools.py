import os
from typing import Tuple
from roboflow import Roboflow


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
        print()
        print("Dataset already downloaded")
        return False

    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise ValueError("ROBOFLOW_API_KEY environment variable not set, add it to .env file.")
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    user, project_name, dataset_version = parse_url(url)

    project = rf.workspace(user).project(project_name)
    project.version(int(dataset_version)).download("yolov8", target_dir)
    return True
