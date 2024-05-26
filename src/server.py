import os
import uuid
import tempfile
import mimetypes
from typing import Any, Dict, Tuple, Sequence, cast

import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from modal import App, asgi_app, Function, Image, gpu, Volume, Dict as ModalDict
from modal.functions import FunctionCall

from pflow.workflow import run_workflow

jobs_volume = Volume.from_name("pflow_jobs", create_if_missing=True)

persisted_jobs_dict: Dict[str, Any] = cast(
    Dict[str, Any], ModalDict.from_name("pflow_jobs_dict", create_if_missing=True)
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

app = App("pflow")

web_app = fastapi.FastAPI()


@app.function(image=image, volumes={"/root/jobs_data": jobs_volume})
@asgi_app()
def fastapi_app() -> Any:
    return web_app


def set_env(env_variables: Dict[str, Any], job_id: str) -> None:
    for key, value in env_variables.items():
        os.environ[key] = value
    if os.environ.get("BASE_FOLDER") is None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["BASE_FOLDER"] = tmp
            print(f"Setting BASE_FOLDER to {tmp}")
    os.environ["PERSISTED_FOLDER"] = f"/root/jobs_data/{job_id}"
    if not os.path.exists(os.environ["PERSISTED_FOLDER"]):
        os.makedirs(os.environ["PERSISTED_FOLDER"], exist_ok=True)


@app.function(image=image, timeout=5_400, volumes={"/root/jobs_data": jobs_volume})
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
    image=image_gpu, gpu=GPU_TYPE, timeout=5_400, volumes={"/root/jobs_data": jobs_volume}
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


@web_app.post("/workflow/gpu")
async def workflow_gpu(request: fastapi.Request) -> Dict[str, Any]:
    request_json = await request.json()
    print("workflow_gpu")
    return start_job(request_json, endpoint_run_workflow_gpu)


@web_app.post("/workflow/cpu")
async def workflow_cpu(request: fastapi.Request) -> Dict[str, Any]:
    request_json = await request.json()
    print("workflow_cpu")
    return start_job(request_json, endpoint_run_workflow_cpu)


@web_app.get("/result/{job_id}")
async def poll_results(job_id: str) -> JSONResponse:
    job_info = persisted_jobs_dict[job_id]
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
