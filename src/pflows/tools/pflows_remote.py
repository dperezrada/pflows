import requests
import os
import json

def prepare_headers(token: str | None) -> dict:
    return {
        'Authorization': f'Bearer {token}' if token else None,
        'Content-Type': 'application/json',
    }

def run(base_url: str, token: str|None, workflow: str|dict, env: dict|None = None, gpu: bool = False) -> dict:
    """
    Run a workflow on the remote server.
    :param base_url: The base URL of the remote server.
    :param token: The token to authenticate the request.
    :param workflow: The workflow to run. Can be the workflow_path as str or a dictionary.
    :param env: The environment variables to pass to the workflow.
    :param gpu: Whether to run the workflow on the GPU.
    :return: The response from the server."""

    headers = prepare_headers(token)
    mode = 'gpu' if gpu else 'cpu'
    url = f'{base_url}/workflow/{mode}'
    response = requests.post(url, headers=headers, data=json.dumps({
        "workflow": workflow,
        "env": env or {},
    }))
    return response.json()


def download_job_file(base_url: str, token: str | None, job_id: str, remote_path: str, local_path: str) -> None:
    headers = prepare_headers(token)

    url = f'{base_url}/download/{job_id}'
    response = requests.post(url, headers=headers, json={"path": remote_path})

    if response.status_code == 200:
        content_type = response.headers.get('Content-Type') or ''

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if content_type == 'application/json':
            data = response.json()
            with open(local_path, 'w') as f:
                json.dump(data, f)
        elif content_type.startswith('text/'):
            with open(local_path, 'w') as f:
                f.write(response.text)
        else:
            with open(local_path, 'wb') as f:
                f.write(response.content)
    else:
        raise Exception(f'Failed to download file: {response.status_code} - {response.text}')


def result(base_url: str, token: str | None, job_id: str) -> dict:
    headers = prepare_headers(token)
    url = f'{base_url}/result/{job_id}'
    response = requests.get(url, headers=headers)
    return response.json()


def upload_file(base_url: str, token: str | None, local_path: str, remote_name: str) -> None:
    headers = prepare_headers(token)
    headers['Content-Type'] = 'multipart/form-data'

    with open(local_path, 'rb') as f:
        url = f'{base_url}/upload'
        response = requests.post(url, headers=headers, json={"path": remote_name}, files={'file': f})

        if response.status_code != 200:
            raise Exception(f'Failed to upload file: {response.status_code} - {response.text}')