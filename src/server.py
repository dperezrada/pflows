import os
import tempfile
from typing import Any, Dict, Sequence, Tuple

import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from modal import App, asgi_app, Image, gpu
from modal.functions import FunctionCall

from pflow.workflow import run_workflow

GPU_TYPE = gpu.A100(size="40GB", count=1)

image = (
    Image
    .debian_slim()
    .apt_install("libgl1-mesa-dev")
    .apt_install("libglib2.0-0")
    .pip_install_from_requirements(
        "./requirements.txt"
    )
)

image_gpu = (
    Image
    .debian_slim()
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements(
        "./requirements.txt", gpu=GPU_TYPE
    )
)

app = App()

web_app = fastapi.FastAPI()


@app.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app


@app.function(image=image)
def endpoint_run_workflow_cpu(workflow: Sequence[Dict[str, Any]], env: Dict[str, Any]):
    print("Running workflow")
    set_env(env)
    results = run_workflow(raw_workflow=workflow)
    json_compatible_results = jsonable_encoder(results)
    return json_compatible_results

@app.function(image=image_gpu, gpu=GPU_TYPE)
def endpoint_run_workflow_gpu(workflow: Sequence[Dict[str, Any]], env: Dict[str, Any]):
    print("Running workflow")
    set_env(env)
    results = run_workflow(raw_workflow=workflow)
    json_compatible_results = jsonable_encoder(results)
    return json_compatible_results


def set_env(env_variables: Dict[str, Any]) -> None:
    for key, value in env_variables.items():
        os.environ[key] = value
    if os.environ.get("BASE_FOLDER") is None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['BASE_FOLDER'] = tmp
            print(f"Setting BASE_FOLDER to {tmp}")

def get_request(request_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    workflow = request_json.get("workflow")
    if workflow is None:
        raise ValueError("The workflow is required.")
    env_variables = request_json.get("env") or {}
    return workflow, env_variables

@web_app.post("/workflow/gpu")
async def workflow_gpu(request: fastapi.Request):
    request_json = await request.json()
    try:
        workflow, env = get_request(request_json)
    except ValueError as e:
        return {"error": str(e)}
    call = endpoint_run_workflow_gpu.spawn(workflow, env)
    if call is None:
        return {"error": "Failed to start workflow"}
    return {"call_id": call.object_id}

@web_app.post("/workflow/cpu")
async def workflow_cpu(request: fastapi.Request):
    request_json = await request.json()
    print("workflow_cpu")
    try:
        workflow, env = get_request(request_json)
    except ValueError as e:
        return {"error": str(e)}
    call = endpoint_run_workflow_cpu.spawn(workflow, env)
    if call is None:
        return {"error": "Failed to start workflow"}
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = FunctionCall.from_id(call_id)
    try:
        return JSONResponse(content=function_call.get(timeout=0))
    except TimeoutError:
        http_accepted_code = 202
        return fastapi.responses.JSONResponse({}, status_code=http_accepted_code)