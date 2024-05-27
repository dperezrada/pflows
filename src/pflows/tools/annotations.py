from typing import List
from pflows.typedef import Annotation


def filter_by_tag(annotations: List[Annotation], tag: str) -> List[Annotation]:
    return [annotation for annotation in annotations if tag in annotation.tags]
