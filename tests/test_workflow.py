import os
import json
import tempfile

from pflows.workflow import read_workflow


workflow_w_variable = [
  {
    "task": "set_env_var",
    "name": "DATA_ID",
    "value": "group3"
  },
  {
    "task": "yolo_v8.load_dataset",
    "folder_path": "{{BASE_FOLDER}}/datasets/downloaded/{{DATA_ID}}"
  }
]

def test_read_w_variable_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'workflow_w_variable.json')
        with open(path, 'w') as f:
            f.write(json.dumps(workflow_w_variable))
        os.environ['BASE_FOLDER'] = tmp
        workflow, workflow_data = read_workflow(workflow_path=path)
        assert len(workflow) == 1
        first_task = workflow[0]
        assert first_task.params['folder_path'] == f"{tmp}/datasets/downloaded/group3"
  

def test_read_w_variable_raw_workflow():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['BASE_FOLDER'] = tmp
        workflow, workflow_data = read_workflow(raw_workflow=workflow_w_variable)
        assert len(workflow) == 1
        first_task = workflow[0]
        assert first_task.params['folder_path'] == f"{tmp}/datasets/downloaded/group3"
