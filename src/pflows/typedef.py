# pylint: disable=R0902

import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass, field, replace

from PIL import Image as PILImage, ImageDraw


def generate_random_color() -> Tuple[int, int, int]:
    """Generate a random RGB color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def relative_to_absolute_coords(coords: Tuple[float, ...], width: int, height: int) -> List[int]:
    """
    Convert relative coordinates to absolute coordinates based on image dimensions.

    Args:
        coords (Tuple[float, ...]): Relative coordinates.
        width (int): Image width.
        height (int): Image height.

    Returns:
        List[int]: Absolute coordinates.
    """
    return [
        int(coords[i] * width if i % 2 == 0 else coords[i] * height) for i in range(len(coords))
    ]


@dataclass
class Category:
    """Represents a category for annotations."""

    id: int
    name: str


@dataclass
class Annotation:
    """Represents an annotation in an image."""

    id: str
    category_id: int
    center: (
        Tuple[float, float] | None
    )  # Format: (x, y) this is relative to the image size (between 0 and 1)
    bbox: (
        Tuple[float, float, float, float] | None
    )  # Format: (x1, y1, x2, y2) this is relative to the image size (between 0 and 1)
    segmentation: (
        Tuple[float, ...] | None
    )  # Format: (x1, y1, x2, y2, ...) this is relative to the image size (between 0 and 1)
    conf: float = -1.0  # Confidence score, between 0 and 1 (default: -1.0)
    category_name: str = ""
    tags: List[str] = field(default_factory=list)
    original_id: Optional[str] = None
    truncated: Optional[bool] = False
    model_id: Optional[str] = None
    obb: Optional[Tuple[float, ...]] = None  # Format (x1, y1, x2, y2, x3, y3, x4, y4)
    task: str = "detect"  # "segment" or "detect" or "obb"

    def distance(self, other_annotation) -> float:
        """
        Calculate the Euclidean distance between the centers of two annotations.

        Args:
            other_annotation (Annotation): Another annotation to compare with.

        Returns:
            float: Distance between the centers, or -1.0 if centers are not available.
        """
        if self.center and other_annotation.center:
            return (
                (self.center[0] - other_annotation.center[0]) ** 2
                + (self.center[1] - other_annotation.center[1]) ** 2
            ) ** 0.5
        return -1.0

    def dict(self) -> Dict[str, Any]:
        """Convert the annotation to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """
        Create an Annotation instance from a dictionary, discarding any fields not in the schema.

        Args:
            data (dict): Dictionary containing annotation data.

        Returns:
            Annotation: An instance of Annotation.
        """
        if isinstance(data, cls):
            return data

        # Get the field names defined in the Annotation class
        valid_fields = cls.__annotations__.keys()

        # Filter the input data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


@dataclass
class Image:
    """Represents an image with annotations."""

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
        """
        Create a shallow copy of the Image instance, including deep copies of mutable fields.

        Returns:
            Image: A copy of the Image instance.
        """
        new_image = replace(self)
        new_image.intermediate_ids = self.intermediate_ids.copy()
        new_image.annotations = [replace(ann) for ann in self.annotations]
        new_image.tags = self.tags.copy()
        new_image.info = self.info.copy()
        return new_image

    def draw(self) -> PILImage.Image:
        """
        Draw annotations on the image.

        Returns:
            PILImage.Image: The image with drawn annotations.
        """
        img = PILImage.open(self.path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        category_colors = {}
        # yellow for text
        text_color = (255, 255, 0)

        for annotation in self.annotations:
            if annotation.category_name not in category_colors:
                category_colors[annotation.category_name] = generate_random_color()

            color = category_colors[annotation.category_name]

            if annotation.task in ["segment", "obb"] and annotation.segmentation:
                abs_segmentation = relative_to_absolute_coords(
                    annotation.segmentation, width, height
                )
                points = list(zip(abs_segmentation[0::2], abs_segmentation[1::2]))
                draw.polygon(points, outline=color, width=3)

            if annotation.task == "detect" and annotation.bbox:
                x1, y1, x2, y2 = annotation.bbox
                abs_x1 = int(x1 * width)
                abs_y1 = int(y1 * height)
                abs_x2 = int(x2 * width)
                abs_y2 = int(y2 * height)
                draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=color, width=2)

            if annotation.center:
                cx, cy = annotation.center
                abs_cx = int(cx * width)
                abs_cy = int(cy * height)
                radius = 3
                draw.ellipse(
                    [abs_cx - radius, abs_cy - radius, abs_cx + radius, abs_cy + radius], fill=color
                )

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
                    continue
                text = annotation.category_name
                if annotation.conf >= 0:
                    text = f"{annotation.category_name} ({annotation.conf:.2f})"
                draw.text(label_position, text, fill=text_color)

        return img

    def dict(self) -> Dict[str, Any]:
        """Convert the image to a dictionary, including annotations."""
        return {**asdict(self), "annotations": [ann.dict() for ann in self.annotations]}

    @classmethod
    def from_dict(cls, data):
        """
        Create an Image instance from a dictionary, discarding any fields not in the schema.

        Args:
            data (dict): Dictionary containing image data.

        Returns:
            Image: An instance of Image.
        """
        if isinstance(data, cls):
            return data

        valid_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        annotations = [Annotation.from_dict(ann) for ann in filtered_data.get("annotations", [])]
        filtered_data["annotations"] = annotations

        return cls(**filtered_data)


@dataclass
class Dataset:
    """Represents a dataset containing images, categories, and groups."""

    images: List[Image]
    categories: List[Category]
    groups: List[str]

    @classmethod
    def from_dict(cls, data):
        """
        Create a Dataset instance from a dictionary, discarding any fields not in the schema.

        Args:
            data (dict): Dictionary containing dataset data.

        Returns:
            Dataset: An instance of Dataset.
        """
        if isinstance(data, cls):
            return data

        valid_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        images = [Image.from_dict(img) for img in filtered_data.get("images", [])]
        categories = [Category(**cat) for cat in filtered_data.get("categories", [])]
        groups = filtered_data.get("groups", [])

        return cls(images=images, categories=categories, groups=groups)


@dataclass
class Task:
    """Represents a task to be performed."""

    task: str
    function: Callable[..., Any]
    params: Dict[str, Any]
    skip: bool = False
    id: str | None = None


@dataclass
class Model:
    """Represents a model to be used."""

    id: str
    name: str
    task: str
    type: str
    categories: List[str]
    params: Dict[str, Any]
    version: str
    size_kb: int
