import os
import uuid
import tempfile
import mimetypes
from functools import wraps
from typing import Any, Dict, Tuple, Sequence, cast

import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from modal import App, asgi_app, Function, Image, gpu, Volume, Dict as ModalDict, Secret
from modal.functions import FunctionCall


from pflows.workflow import run_workflow

jobs_volume = Volume.from_name("pflows_jobs", create_if_missing=True)
uploads_volume = Volume.from_name("pflows_upload", create_if_missing=True)

persisted_jobs_dict: Dict[str, Any] = cast(
    Dict[str, Any], ModalDict.from_name("pflows_jobs_dict", create_if_missing=True)
)

GPU_TYPE = gpu.A100(size="40GB", count=1)

image = (
    Image.debian_slim()
    .apt_install("libgl1-mesa-dev")
    .apt_install("libglib2.0-0")
    .pip_install_from_requirements("./requirements.txt")
)

image_gpu = (
    Image.debian_slim()
    .apt_install(["ffmpeg", "libsm6", "libxext6"])
    .pip_install_from_requirements("./requirements.txt", gpu=GPU_TYPE)
)

app = App("pflows")

web_app = fastapi.FastAPI()


@app.function(
    image=image,
    volumes={"/root/jobs_data": jobs_volume, "/root/uploads_data": uploads_volume},
    secrets=[Secret.from_name("PFLOW_AUTH_TOKEN")],
    timeout=600,
)
@asgi_app()
def fastapi_app() -> Any:
    return web_app


def set_env(env_variables: Dict[str, Any], job_id: str) -> None:
    for key, value in env_variables.items():
        os.environ[key] = value
    if os.environ.get("BASE_FOLDER") is None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["BASE_FOLDER"] = tmp
            os.environ["REMOTE_BASE_FOLDER"] = os.environ["BASE_FOLDER"]
            print(f"Setting BASE_FOLDER to {tmp}")
    os.environ["JOB_ID"] = job_id

    os.environ["PERSISTED_FOLDER"] = f"/root/jobs_data/{job_id}"
    os.environ["REMOTE_PERSISTED_FOLDER"] = os.environ["PERSISTED_FOLDER"]
    if not os.path.exists(os.environ["PERSISTED_FOLDER"]):
        os.makedirs(os.environ["PERSISTED_FOLDER"], exist_ok=True)
    os.environ["TEMP_FOLDER"] = tempfile.gettempdir()
    os.environ["REMOTE_TEMP_FOLDER"] = os.environ["TEMP_FOLDER"]


@app.function(
    image=image,
    timeout=7 * 60 * 60,
    volumes={"/root/jobs_data": jobs_volume, "/root/uploads_data": uploads_volume},
)
def endpoint_run_workflow_cpu(
    workflow: Sequence[Dict[str, Any]], env: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    print("Running workflow")
    set_env(env, job_id)
    results = run_workflow(
        raw_workflow=workflow, store_dict=persisted_jobs_dict, store_dict_key=job_id
    )
    persisted_jobs_dict[job_id] = {**results, "status": "completed"}
    return cast(Dict[str, Any], persisted_jobs_dict[job_id])


@app.function(
    image=image_gpu,
    gpu=GPU_TYPE,
    timeout=7 * 60 * 60,
    volumes={"/root/jobs_data": jobs_volume, "/root/uploads_data": uploads_volume},
)
def endpoint_run_workflow_gpu(
    workflow: Sequence[Dict[str, Any]], env: Dict[str, Any], job_id: str
) -> Dict[str, Any]:
    print("Running workflow")
    set_env(env, job_id)
    results = run_workflow(
        raw_workflow=workflow, store_dict=persisted_jobs_dict, store_dict_key=job_id
    )
    persisted_jobs_dict[job_id] = {**results, "status": "completed"}
    return cast(Dict[str, Any], persisted_jobs_dict[job_id])


def get_request(request_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    workflow = request_json.get("workflow")
    if workflow is None:
        raise ValueError("The workflow is required.")
    env_variables = request_json.get("env") or {}
    return workflow, env_variables


def start_job(request_json: Dict[str, Any], endpoint: Function) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    try:
        workflow, env = get_request(request_json)
    except ValueError as e:
        return {"error": str(e)}
    call = endpoint.spawn(workflow, env, job_id)
    if call is None:
        return {"error": "Failed to start workflow"}
    persisted_jobs_dict[job_id] = {"call_id": call.object_id, "status": "running"}
    return {"call_id": call.object_id, "job_id": job_id, "status": "running"}


def auth_required(endpoint_func: Any) -> Any:
    @wraps(endpoint_func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        equal_tokens = False
        try:
            request: fastapi.Request | None = (
                kwargs.get("request") or kwargs.get("_request") or None
            )
            if request is None:
                raise ValueError("Request not found")
            token = (request.headers.get("authorization") or "").split(" ")[1]
            equal_tokens = os.environ.get("PFLOW_AUTH_TOKEN") == token
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        if not equal_tokens:
            raise HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await endpoint_func(*args, **kwargs)

    return wrapper


@web_app.post("/workflow/gpu")
@auth_required
async def workflow_gpu(request: fastapi.Request) -> Dict[str, Any]:
    request_json = await request.json()
    print("workflow_gpu")
    return start_job(request_json, endpoint_run_workflow_gpu)


@web_app.post("/workflow/cpu")
@auth_required
async def workflow_cpu(request: fastapi.Request) -> Dict[str, Any]:
    request_json = await request.json()
    print("workflow_cpu")
    return start_job(request_json, endpoint_run_workflow_cpu)


@web_app.get("/result/{job_id}")
@auth_required
async def poll_results(_request: fastapi.Request, job_id: str) -> Any:
    try:
        job_info = persisted_jobs_dict[job_id]
    except KeyError:
        return {"error": "Job not found."}
    if job_info.get("status") == "completed":
        return JSONResponse(content=jsonable_encoder(persisted_jobs_dict[job_id]))
    function_call = FunctionCall.from_id(job_info["call_id"])
    try:
        json_result = function_call.get(timeout=0)
        persisted_jobs_dict[job_id] = {
            **json_result,
            **persisted_jobs_dict[job_id],
            "status": "completed",
        }
        return JSONResponse(content=jsonable_encoder(persisted_jobs_dict[job_id]))

    except TimeoutError:
        http_accepted_code = 202
        return fastapi.responses.JSONResponse(
            content=jsonable_encoder(job_info), status_code=http_accepted_code
        )


@web_app.post("/download/{job_id}")
@auth_required
async def download_data(request: fastapi.Request, job_id: str) -> Any:
    request_path = (await request.json()).get("path")
    print("request_path", request_path)
    if request_path is None:
        return {"error": "The path is required."}
    request_path = request_path.lstrip("/")

    sanitized_path = os.path.normpath(request_path)

    if ".." in sanitized_path:
        return {"error": "Invalid path."}

    abs_path = f"/root/jobs_data/{job_id}/{request_path}"

    print(abs_path)
    if not os.path.exists(abs_path):
        return {"error": "File not found."}

    content_type, _ = mimetypes.guess_type(abs_path)
    if content_type is None:
        content_type = "application/octet-stream"  # Default content type if unknown

    # Read the file content
    with open(abs_path, "rb") as file:
        file_content = file.read()

    # Return the file content with the guessed content type
    return fastapi.Response(content=file_content, media_type=content_type)


@web_app.post("/uploads/download/{upload_id}")
@auth_required
async def download_upload_file(request: fastapi.Request, upload_id: str) -> Any:
    request_path = (await request.json()).get("path")
    print("request_path", request_path)
    if request_path is None:
        return {"error": "The path is required."}
    request_path = request_path.lstrip("/")

    sanitized_path = os.path.normpath(request_path)

    if ".." in sanitized_path:
        return {"error": "Invalid path."}

    abs_path = f"/root/uploads_data/{upload_id}/{request_path}"

    if not os.path.exists(abs_path):
        return {"error": "File not found."}

    content_type, _ = mimetypes.guess_type(abs_path)
    if content_type is None:
        content_type = "application/octet-stream"

    # Read the file content
    with open(abs_path, "rb") as file:
        file_content = file.read()

    # Return the file content with the guessed content type
    return fastapi.Response(content=file_content, media_type=content_type)


@web_app.post("/uploads")
@auth_required
async def upload_file(request: fastapi.Request) -> Any:
    data = await request.form()
    file = data["file"]
    path = data["path"]
    upload_id = uuid.uuid4().hex
    abs_path = f"/root/uploads_data/{upload_id}/{path}"
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if isinstance(file, str):
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(file)
    else:
        with open(abs_path, "wb") as f:
            f.write(await file.read())

    return {
        "status": "uploaded",
        "path": path,
        "upload_id": upload_id,
        "size": os.path.getsize(abs_path),
        "abs_path": abs_path,
    }


@web_app.post("/uploads/delete/{upload_id}")
@auth_required
async def delete_upload_file(request: fastapi.Request, upload_id: str) -> Any:
    request_path = (await request.json()).get("path") or ""
    if request_path is None:
        return {"error": "The path is required."}

    request_path = request_path.split("/")[-1]

    sanitized_path = os.path.normpath(request_path)

    if ".." in sanitized_path:
        return {"error": "Invalid path."}

    abs_path = f"/root/uploads_data/{upload_id}/{request_path}"

    print(abs_path)
    if not os.path.exists(abs_path):
        return {"error": "File not found."}

    os.remove(abs_path)
    return {"status": "deleted"}
