# pylint: disable=R0902

from dataclasses import dataclass, field

from typing import List, Tuple, TypedDict


@dataclass
class Category:
    id: int
    name: str


@dataclass
class Annotation:
    id: int
    category_id: int
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    segmentation: Tuple[float]
    task: str
    conf: float = -1.0
    category_name: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Image:
    id: int
    path: str
    intermediate_ids: List[int]
    width: int
    height: int
    size_kb: int
    group: str
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class Dataset:
    images: List[Image]
    categories: List[Category]


class DatasetDict(TypedDict):
    dataset: Dataset
