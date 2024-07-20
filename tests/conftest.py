import warnings
from distutils.version import LooseVersion

from pflows.typedef import Dataset
warnings.filterwarnings("ignore", category=DeprecationWarning, module="thop")

import pytest 
from pathlib import Path

from pflows.tools.yolo_v8 import load_dataset

current_folder = Path(__file__).parent

@pytest.fixture
def dataset():
    return load_dataset(
        Dataset(
            images=[],
            categories=[],
            groups=[],
        ),
        str(current_folder / "fixtures/CUB200_parts.v24-070.green_violetear.yolov8")
    )