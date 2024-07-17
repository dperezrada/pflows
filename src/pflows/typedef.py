# pylint: disable=R0902

import random
from dataclasses import asdict, dataclass, field, replace

from PIL import Image as PILImage, ImageDraw

from typing import Any, Callable, Dict, List, Optional, Tuple


def generate_random_color() -> Tuple[int, int, int]:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def relative_to_absolute_coords(coords: Tuple[float, ...], width: int, height: int) -> List[int]:
    return [
        int(coords[i] * width if i % 2 == 0 else coords[i] * height) for i in range(len(coords))
    ]


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
    original_id: Optional[str] = None
    truncated: Optional[bool] = False

    def distance(self, other_annotation) -> float:
        if self.center and other_annotation.center:
            return (
                (self.center[0] - other_annotation.center[0]) ** 2
                + (self.center[1] - other_annotation.center[1]) ** 2
            ) ** 0.5
        return -1.0

    def dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, cls):
            return data
        return cls(**data)


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
    info: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        # Create a shallow copy of the dataclass
        new_image = replace(self)

        # Explicitly copy mutable fields
        new_image.intermediate_ids = self.intermediate_ids.copy()
        new_image.annotations = [replace(ann) for ann in self.annotations]
        new_image.tags = self.tags.copy()
        new_image.info = self.info.copy()

        return new_image

    def draw(self) -> PILImage.Image:
        # Load the image
        img = PILImage.open(self.path)
        draw = ImageDraw.Draw(img)
        width, height = img.size

        for annotation in self.annotations:
            color = generate_random_color()

            if annotation.task == "segment" and annotation.segmentation:
                # Convert relative segmentation coordinates to absolute
                abs_segmentation = relative_to_absolute_coords(
                    annotation.segmentation, width, height
                )
                # Draw segmentation outline
                points = list(zip(abs_segmentation[0::2], abs_segmentation[1::2]))
                draw.polygon(points, outline=color)  # Remove fill parameter

            if annotation.task == "detect" and annotation.bbox:
                # Convert relative bbox coordinates to absolute
                x1, y1, x2, y2 = annotation.bbox
                abs_x1 = int(x1 * width)
                abs_y1 = int(y1 * height)
                abs_x2 = int(x2 * width)
                abs_y2 = int(y2 * height)

                # Draw bounding box
                draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=color, width=2)

            # Draw center point if available
            if annotation.center:
                cx, cy = annotation.center
                abs_cx = int(cx * width)
                abs_cy = int(cy * height)
                radius = 3
                draw.ellipse(
                    [abs_cx - radius, abs_cy - radius, abs_cx + radius, abs_cy + radius], fill=color
                )

            # Add label if category_name is available
            if annotation.category_name:
                if annotation.bbox:
                    label_position = (
                        int(annotation.bbox[0] * width),
                        int(annotation.bbox[1] * height) - 15,
                    )
                elif annotation.center:
                    label_position = (
                        int(annotation.center[0] * width),
                        int(annotation.center[1] * height),
                    )
                else:
                    continue  # Skip label if no position available
                text = annotation.category_name
                if annotation.conf >= 0:
                    text = f"{annotation.category_name} ({annotation.conf:.2f})"
                draw.text(label_position, text, fill=color)

        return img

    def dict(self) -> Dict[str, Any]:
        return {**asdict(self), "annotations": [ann.dict() for ann in self.annotations]}

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, cls):
            return data
        annotations: List[Annotation] = [
            Annotation.from_dict(ann) for ann in data.get("annotations", [])
        ]
        return cls(**data, annotations=annotations)


@dataclass
class Dataset:
    images: List[Image]
    categories: List[Category]
    groups: List[str]

    @classmethod
    def from_dict(cls, data):
        raw_images = data.images if hasattr(data, "images") else data.get("images", [])
        images = [Image.from_dict(img) for img in raw_images]
        categories = (
            data.categories
            if hasattr(data, "categories")
            else [Category(**cat) for cat in data.get("categories", [])]
        )
        groups = data.groups if hasattr(data, "groups") else data.get("groups", [])
        return cls(images=images, categories=categories, groups=groups)


@dataclass
class Task:
    task: str
    function: Callable[..., Any]
    params: Dict[str, Any]
    skip: bool = False
    id: str | None = None
