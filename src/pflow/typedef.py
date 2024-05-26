# pylint: disable=R0902

from dataclasses import dataclass, field

from typing import Any, Callable, Dict, List, Tuple


@dataclass
class Category:
    id: int
    name: str


@dataclass
class Annotation:
    id: str
    category_id: int
    center: Tuple[float, float] | None
    bbox: Tuple[float, float, float, float] | None
    segmentation: Tuple[float, ...] | None
    task: str
    conf: float = -1.0
    category_name: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Image:
    id: str
    path: str
    intermediate_ids: List[str]
    width: int
    height: int
    size_kb: int
    group: str
    annotations: List[Annotation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Dataset:
    images: List[Image]
    categories: List[Category]
    groups: List[str]


@dataclass
class Task:
    task: str
    function: Callable[..., Any]
    params: Dict[str, Any]
    skip: bool = False
    id: str | None = None
