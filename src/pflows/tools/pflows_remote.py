import os
import json
from typing import Dict, Any, cast

import requests


class PFlowsRemoteError(Exception):
    pass


def prepare_headers(token: str | None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def run(
    base_url: str,
    token: str | None,
    workflow: str | Dict[str, Any],
    env: Dict[str, str] | None = None,
    gpu: bool = False,
) -> Dict[str, Any]:
    """
    Run a workflow on the remote server.
    :param base_url: The base URL of the remote server.
    :param token: The token to authenticate the request.
    :param workflow: The workflow to run. Can be the workflow_path as str or a dictionary.
    :param env: The environment variables to pass to the workflow.
    :param gpu: Whether to run the workflow on the GPU.
    :return: The response from the server."""

    headers = prepare_headers(token)
    mode = "gpu" if gpu else "cpu"
    url = f"{base_url}/workflow/{mode}"
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(
            {
                "workflow": workflow,
                "env": env or {},
            }
        ),
        timeout=60 * 60 * 2,
    )
    return cast(Dict[str, Any], response.json())


def download_job_file(
    base_url: str, token: str | None, job_id: str, remote_path: str, local_path: str
) -> None:
    headers = prepare_headers(token)

    url = f"{base_url}/download/{job_id}"
    response = requests.post(
        url,
        headers=headers,
        json={"path": remote_path},
        timeout=60 * 30,
    )

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type") or ""

        target_dir = os.path.dirname(local_path)
        if target_dir != "":
            os.makedirs(target_dir, exist_ok=True)
        if content_type == "application/json":
            data = response.json()
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        elif content_type.startswith("text/"):
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            with open(local_path, "wb") as f:
                f.write(response.content)
    else:
        raise PFlowsRemoteError(
            f"Failed to download file: {response.status_code} - {response.text}"
        )


def result(base_url: str, token: str | None, job_id: str) -> Dict[str, Any]:
    headers = prepare_headers(token)
    url = f"{base_url}/result/{job_id}"
    response = requests.get(url, headers=headers, timeout=60 * 10)
    return cast(Dict[str, Any], response.json())


def upload_file(
    base_url: str, token: str | None, local_path: str, remote_name: str
) -> Dict[str, Any]:
    headers = prepare_headers(token)
    del headers["Content-Type"]

    with open(local_path, "rb") as f:
        url = f"{base_url}/uploads"
        files = {"file": (remote_name, f)}
        data = {"path": remote_name}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=60 * 60)
        if response.status_code != 200:
            raise PFlowsRemoteError(
                f"Failed to upload file: {response.status_code} - {response.text}"
            )
        return cast(Dict[str, Any], response.json())


def download_upload_file(
    base_url: str, token: str | None, upload_id: str, remote_path: str, local_path: str
) -> None:
    headers = prepare_headers(token)

    url = f"{base_url}/uploads/download/{upload_id}"
    response = requests.post(url, headers=headers, json={"path": remote_path}, timeout=60 * 30)

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type") or ""

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if content_type == "application/json":
            data = response.json()
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        elif content_type.startswith("text/"):
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            with open(local_path, "wb") as f:
                f.write(response.content)
    else:
        raise PFlowsRemoteError(
            f"Failed to download file: {response.status_code} - {response.text}"
        )
