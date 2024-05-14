from typing import List
from pflow.typedef import Annotation


def filter_by_tag(annotations: List[Annotation], tag: str) -> List[Annotation]:
    return [annotation for annotation in annotations if tag in annotation.tags]
