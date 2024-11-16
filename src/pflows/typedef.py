# pylint: disable=R0902

import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass, field, replace

from PIL import Image as PILImage, ImageDraw, ImageFont


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
        Create an Annotation instance from a dictionary, including fields from parent classes.

        Args:
            data (dict): Dictionary containing annotation data.

        Returns:
            Annotation: An instance of Annotation or its subclass.
        """
        if isinstance(data, cls):
            return data

        # Get all fields from the current class and its parent classes
        valid_fields = set()
        for c in cls.__mro__:
            if hasattr(c, "__annotations__"):
                valid_fields.update(c.__annotations__.keys())

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

    def draw(self, show_conf: bool = False) -> PILImage.Image:
        """
        Draw annotations on the image with:
        - Segmentation: semi-transparent fill
        - Detection: dotted border only (no fill)
        """
        img = PILImage.open(self.path)
        img = img.convert("RGB")
        overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size
        # Make font size proportional to the smaller image dimension (1.5% of smaller dimension)
        font_size = int(min(width, height) * 0.015)
        font = ImageFont.load_default(size=font_size)

        # First pass: Draw all annotations (segmentations and bounding boxes)
        for annotation in self.annotations:
            # Generate color similar to web version
            hue = (annotation.category_id * 137.508) % 360
            h = hue / 360
            rgb_color = self._hsl_to_rgb(h, 0.7, 0.5)

            if annotation.task == "segment" and annotation.segmentation:
                # Segmentation gets fill and outline
                fill_color = (*rgb_color, 51)  # 20% opacity
                stroke_color = (*rgb_color, 25)  # 10% opacity

                abs_segmentation = relative_to_absolute_coords(
                    annotation.segmentation, width, height
                )
                points = list(zip(abs_segmentation[0::2], abs_segmentation[1::2]))
                draw.polygon(points, fill=fill_color, outline=stroke_color)

            elif annotation.task == "detect" and annotation.bbox:
                # Detection gets only dotted outline, no fill
                stroke_color = (*rgb_color, 255)  # Full opacity for border
                x1, y1, x2, y2 = annotation.bbox
                abs_coords = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]

                # Draw dotted rectangle (manually since PIL doesn't support dotted lines)
                dash_length = 5
                for i in range(0, int(x2 * width - x1 * width), dash_length * 2):
                    # Top line
                    draw.line(
                        [
                            abs_coords[0] + i,
                            abs_coords[1],
                            min(abs_coords[0] + i + dash_length, abs_coords[2]),
                            abs_coords[1],
                        ],
                        fill=stroke_color,
                    )
                    # Bottom line
                    draw.line(
                        [
                            abs_coords[0] + i,
                            abs_coords[3],
                            min(abs_coords[0] + i + dash_length, abs_coords[2]),
                            abs_coords[3],
                        ],
                        fill=stroke_color,
                    )

                for i in range(0, int(y2 * height - y1 * height), dash_length * 2):
                    # Left line
                    draw.line(
                        [
                            abs_coords[0],
                            abs_coords[1] + i,
                            abs_coords[0],
                            min(abs_coords[1] + i + dash_length, abs_coords[3]),
                        ],
                        fill=stroke_color,
                    )
                    # Right line
                    draw.line(
                        [
                            abs_coords[2],
                            abs_coords[1] + i,
                            abs_coords[2],
                            min(abs_coords[1] + i + dash_length, abs_coords[3]),
                        ],
                        fill=stroke_color,
                    )

        # Create a dictionary to store used text positions
        used_positions = {}  # Format: {x_position: [y_positions]}

        # Second pass: Draw all text labels
        for annotation in self.annotations:
            if annotation.category_name:
                # Generate color similar to web version
                hue = (annotation.category_id * 137.508) % 360
                h = hue / 360
                rgb_color = self._hsl_to_rgb(h, 0.7, 0.5)

                # Calculate the center point for text placement
                if annotation.bbox:
                    # Place text at the center-top of the bbox
                    text_x = int((annotation.bbox[0] + annotation.bbox[2]) * width / 2)
                    text_y = int(annotation.bbox[1] * height) - font_size - 2
                elif annotation.segmentation:
                    # Calculate centroid of the segmentation polygon
                    x_coords = annotation.segmentation[0::2]
                    y_coords = annotation.segmentation[1::2]
                    text_x = int(sum(x_coords) / len(x_coords) * width)
                    text_y = int(min(y_coords) * height) - font_size - 2
                else:
                    continue

                text = annotation.category_name
                if annotation.conf >= 0 and show_conf:
                    text += f" ({int(annotation.conf * 100)}%)"

                # Get text dimensions
                text_bbox = draw.textbbox((text_x, text_y), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Center the text horizontally
                text_x -= text_width // 2

                # Ensure text stays within image bounds
                text_x = max(0, min(text_x, width - text_width))
                text_y = max(0, text_y)

                # Check for overlaps in a smaller region around the text
                x_range = range(max(0, text_x), min(width, text_x + text_width))

                # Find all y positions used in this x range
                used_y_positions = set()
                for x in x_range:
                    if x in used_positions:
                        used_y_positions.update(used_positions[x])

                # Find the closest non-overlapping position
                original_y = text_y
                while any(abs(text_y - used_y) < text_height + 2 for used_y in used_y_positions):
                    if text_y < original_y:
                        text_y = original_y + (original_y - text_y) + text_height + 2
                    else:
                        text_y += text_height + 2

                # Record the used position
                for x in x_range:
                    if x not in used_positions:
                        used_positions[x] = []
                    used_positions[x].append(text_y)

                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

        # Composite the transparent overlay onto the original image
        img = PILImage.alpha_composite(img.convert("RGBA"), overlay)
        return img.convert("RGB")

    def _hsl_to_rgb(self, h, s, l):
        """
        Convert HSL to RGB color.
        h, s, l values are in range [0, 1]
        Returns RGB values in range [0, 255]
        """

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return (int(r * 255), int(g * 255), int(b * 255))

    def dict(self) -> Dict[str, Any]:
        """Convert the image to a dictionary, including annotations."""
        return {**asdict(self), "annotations": [ann.dict() for ann in self.annotations]}

    @classmethod
    def from_dict(cls, data):
        """
        Create an Image instance from a dictionary, including fields from parent classes.

        Args:
            data (dict): Dictionary containing image data.

        Returns:
            Image: An instance of Image or its subclass.
        """
        if isinstance(data, cls):
            return data

        valid_fields = set()
        for c in cls.__mro__:
            if hasattr(c, "__annotations__"):
                valid_fields.update(c.__annotations__.keys())

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

        valid_fields = set()
        for c in cls.__mro__:
            if hasattr(c, "__annotations__"):
                valid_fields.update(c.__annotations__.keys())

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
