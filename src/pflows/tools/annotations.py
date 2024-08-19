from typing import List, Sequence, Tuple
from dataclasses import asdict
from scipy.spatial import ConvexHull

import cv2
import numpy as np

from scipy.interpolate import splprep, splev
from ultralytics import YOLO

from pflows.polygons import iou_polygons
from pflows.tools.categories import remap_category_ids
from pflows.typedef import Annotation, Category, Dataset, Image


def filter_by_tag(
    dataset: Dataset, include: List[str] | None = None, exclude: List[str] | None = None
) -> Dataset:
    include = include or []
    exclude = exclude or []

    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        annotation
                        for annotation in image.annotations
                        if (
                            len(include) == 0
                            or any(tag in (annotation.tags or []) for tag in include)
                        )
                        and (
                            len(exclude) == 0
                            or all(tag not in (annotation.tags or []) for tag in exclude)
                        )
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def keep_certain_categories(dataset: Dataset, categories: List[str]) -> Dataset:
    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        annotation
                        for annotation in image.annotations
                        if annotation.category_name in categories
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def new_annotation(annotation, overwrite: dict):
    return Annotation(
        **{
            **asdict(annotation),
            **overwrite,
        }
    )


def change_all_categories(dataset: Dataset, new_category: str) -> Dataset:
    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        new_annotation(
                            annotation, {"category_name": new_category, "category_id": 0}
                        )
                        for annotation in image.annotations
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=[Category(id=0, name=new_category)],
        groups=dataset.groups,
    )


def remove_annotations(
    dataset: Dataset, remove_tags: List[str], remove_categories_model: str | None = None
) -> Dataset:
    categories = dataset.categories
    if remove_categories_model:
        model = model = YOLO(remove_categories_model)
        model_categories = model.names
        if isinstance(model_categories, dict):
            model_categories = model_categories.values()
        categories = [category for category in categories if category.name not in model_categories]
    return remap_category_ids(
        Dataset(
            images=[
                Image(
                    **{
                        **asdict(image),
                        "annotations": [
                            annotation
                            for annotation in image.annotations
                            if all(tag not in (annotation.tags or []) for tag in remove_tags)
                        ],
                    }
                )
                for image in dataset.images
            ],
            categories=categories,
            groups=dataset.groups,
        )
    )


def chaikin_smooth(raw_polygon, iterations=5, tension=0.75):
    # Convert raw_polygon to [(x1, y1), (x2, y2), ...]
    polygon = [(raw_polygon[i], raw_polygon[i + 1]) for i in range(0, len(raw_polygon), 2)]

    for _ in range(iterations):
        smoothed = []
        for i in range(len(polygon)):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % len(polygon)]

            # Create two new points using the tension parameter
            smoothed.append((tension * x0 + (1 - tension) * x1, tension * y0 + (1 - tension) * y1))
            smoothed.append(((1 - tension) * x0 + tension * x1, (1 - tension) * y0 + tension * y1))

        polygon = smoothed

    # Convert back to the format of [x1, y1, x2, y2, ...]
    return [coord for point in polygon for coord in point]


def b_spline_smoothing(polygon, smoothness=0.0005, num_points=70):
    # Convertir el polígono a un array numpy
    points = np.array(polygon).reshape(-1, 2)

    # Cerrar el polígono si no lo está
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))

    # Aplicar spline paramétrico
    tck, u = splprep(points.T, u=None, s=smoothness, per=1)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)

    # Convertir de vuelta a la forma original
    smoothed_polygon = list(zip(x_new, y_new))

    # flatten
    smoothed_polygon = [coord for point in smoothed_polygon for coord in point]
    return smoothed_polygon


def smooth_segments(
    dataset: Dataset,
    technique="spline",
    smoothness: float = 0.0005,
    num_points: int = 70,
    smooth_iterations: int = 1,
    tension: float = 0.75,
) -> Dataset:
    if technique == "spline":
        smooth_polygon = b_spline_smoothing
        args = {"smoothness": smoothness, "num_points": num_points}
    elif technique == "chaikin":
        smooth_polygon = chaikin_smooth
        args = {"iterations": smooth_iterations, "tension": tension}
    else:
        raise ValueError("Invalid smoothing technique")
    return Dataset(
        images=[
            Image(
                **{
                    **asdict(image),
                    "annotations": [
                        (
                            new_annotation(
                                annotation,
                                {"segmentation": smooth_polygon(annotation.segmentation, **args)},
                            )
                            if annotation.task == "segment" and annotation.segmentation
                            else annotation
                        )
                        for annotation in image.annotations
                    ],
                }
            )
            for image in dataset.images
        ],
        categories=dataset.categories,
        groups=dataset.groups,
    )


def smooth_image(image: Image, smoothness=0.0001, num_points=70):
    for annotation in image.annotations:
        if annotation.task == "segment":
            smoothed_polygon = smooth_polygon(annotation.segmentation, smoothness, num_points)
            annotation.segmentation = smoothed_polygon
    return image


def is_complete_annotation(
    annotation: Annotation, image_width: int, image_height: int, border_threshold=10
) -> bool:
    if not annotation or annotation.segmentation is None or annotation.bbox is None:
        return False

    if annotation.bbox:
        x1, y1, x2, y2 = annotation.bbox
        # adjusted x1, y1, x2, y2
        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height
        return (
            border_threshold < x1 < x2 < image_width - border_threshold
            and border_threshold < y1 < y2 < image_height - border_threshold
        )

    points = [
        (point[0] * image_width, point[1] * image_height)
        for point in np.array(annotation.segmentation).reshape(-1, 2)
    ]

    # Create a mask for this segment
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        contour = contours[0]
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is away from the image border
        if (
            x > border_threshold
            and y > border_threshold
            and x + w < image_width - border_threshold
            and y + h < image_height - border_threshold
        ):

            # Check aspect ratio
            aspect_ratio = h / w
            if 1.5 < aspect_ratio < 5:  # Adjust these values as needed

                # Additional check: ensure the segment doesn't touch the bottom
                if np.max(points[:, 1]) < image_height - border_threshold:
                    return True
    return False


def select_image_complete_annotation(image: Image, border_tolerance: int = 10) -> Image:
    complete_annotations = [
        ann
        for ann in image.annotations
        if is_complete_annotation(ann, image.width, image.height, border_tolerance)
    ]
    return Image(
        **{
            **asdict(image),
            "annotations": complete_annotations,
        }
    )


def select_complete_annotations(dataset: Dataset, border_tolerance: int = 10) -> Dataset:
    new_images = []
    for image in dataset.images:
        new_images.append(select_image_complete_annotation(image, border_tolerance))

    return Dataset(images=new_images, categories=dataset.categories, groups=dataset.groups)


def remove_duplicate_annotations(dataset: Dataset, iou_threshold: float = 0.5) -> Dataset:
    new_images = []
    for image in dataset.images:
        new_annotations = []
        for i, annotation in enumerate(image.annotations):
            if annotation.segmentation is None:
                new_annotations.append(annotation)
                continue

            is_duplicate = False
            for other_annotation in image.annotations[i + 1 :]:
                if other_annotation.segmentation is None:
                    continue

                iou = iou_polygons(annotation.segmentation, other_annotation.segmentation)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                new_annotations.append(annotation)

        new_images.append(Image(**{**asdict(image), "annotations": new_annotations}))

    return Dataset(images=new_images, categories=dataset.categories, groups=dataset.groups)


def calculate_bbox_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection / (area1 + area2 - intersection)
    return iou


def replace_category_on_overlap(
    dataset: Dataset,
    search_category: str,
    replace_categories: List[str],
    new_category: str,
    iou_threshold: float = 0.5,
) -> Dataset:
    new_categories = dataset.categories
    if new_category not in [cat.name for cat in dataset.categories]:
        new_categories = dataset.categories + [
            Category(id=len(dataset.categories), name=new_category)
        ]

    def process_image(image: Image) -> Image:
        search_annotations = [
            ann for ann in image.annotations if ann.category_name == search_category
        ]
        replace_annotations = [
            ann for ann in image.annotations if ann.category_name in replace_categories
        ]
        other_annotations = [
            ann
            for ann in image.annotations
            if ann.category_name not in [search_category] + replace_categories
        ]

        new_annotations = []
        for replace_ann in replace_annotations:
            overlap_found = False
            for search_ann in search_annotations:
                if replace_ann.segmentation and search_ann.segmentation:
                    iou = iou_polygons(replace_ann.segmentation, search_ann.segmentation)
                    if iou > iou_threshold:
                        overlap_found = True
                        break
                elif replace_ann.bbox and search_ann.bbox:
                    iou = calculate_bbox_iou(replace_ann.bbox, search_ann.bbox)
                    if iou > iou_threshold:
                        overlap_found = True
                        break

            if overlap_found:
                new_annotations.append(
                    Annotation(
                        id=replace_ann.id,
                        category_id=next(
                            cat.id for cat in new_categories if cat.name == new_category
                        ),
                        category_name=new_category,
                        center=replace_ann.center,
                        bbox=replace_ann.bbox,
                        segmentation=replace_ann.segmentation,
                        task=replace_ann.task,
                        conf=replace_ann.conf,
                        tags=replace_ann.tags,
                    )
                )
            else:
                new_annotations.append(replace_ann)

        # Add all search_category annotations and other annotations
        new_annotations.extend(search_annotations)
        new_annotations.extend(other_annotations)

        return Image(**{**image.__dict__, "annotations": new_annotations})

    new_images = [process_image(image) for image in dataset.images]

    return Dataset(images=new_images, categories=new_categories, groups=dataset.groups)


def change_annotation_from_another_annotation(
    dataset: Dataset,
    search_annotation_category_names: Sequence[str],
    replace_annotation_category_name: Sequence[str],
) -> Dataset:
    for image in dataset.images:
        tooth_numbering_annotations = [
            annotation
            for annotation in image.annotations
            if annotation.category_name in replace_annotation_category_name
        ]
        segmentation_annotations = [
            annotation
            for annotation in image.annotations
            if annotation.category_name in search_annotation_category_names
        ]

        for annotation in segmentation_annotations:
            # find closest center from tooth_numbering_annotations
            closest = None
            min_distance = float("inf")
            for tooth_numbering_annotation in tooth_numbering_annotations:
                distance = annotation.distance(tooth_numbering_annotation)
                if distance < min_distance:
                    min_distance = distance
                    closest = tooth_numbering_annotation
            if closest:
                annotation.category_id = closest.category_id
                annotation.category_name = closest.category_name

    return dataset


def polygon_to_obb(polygon):
    if polygon is None:
        return None
    if len(polygon) < 3:
        return None
    # Ensure polygon is a numpy array
    polygon = np.array(polygon).reshape(-1, 2)

    # Get the convex hull of the polygon
    hull = ConvexHull(polygon)
    hull_points = polygon[hull.vertices]

    # Function to calculate the area of a rectangle
    def rect_area(rect):
        return np.linalg.norm(rect[1] - rect[0]) * np.linalg.norm(rect[2] - rect[1])

    # Initialize variables
    min_rect = None
    min_area = float("inf")

    # Iterate through all edges of the convex hull
    for i in range(len(hull_points)):
        edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
        edge_norm = edge / np.linalg.norm(edge)

        # Create orthogonal vector
        orth = np.array([-edge_norm[1], edge_norm[0]])

        # Project all points onto the edge and its orthogonal
        proj_edge = np.dot(hull_points - hull_points[i], edge_norm)
        proj_orth = np.dot(hull_points - hull_points[i], orth)

        # Find min and max projections
        min_edge, max_edge = np.min(proj_edge), np.max(proj_edge)
        min_orth, max_orth = np.min(proj_orth), np.max(proj_orth)

        # Calculate rectangle vertices
        rect = np.array(
            [
                hull_points[i] + min_edge * edge_norm + min_orth * orth,
                hull_points[i] + max_edge * edge_norm + min_orth * orth,
                hull_points[i] + max_edge * edge_norm + max_orth * orth,
                hull_points[i] + min_edge * edge_norm + max_orth * orth,
            ]
        )

        # Update minimum area rectangle if necessary
        area = rect_area(rect)
        if area < min_area:
            min_area = area
            min_rect = rect

    # Flatten the rectangle coordinates
    return min_rect.flatten().tolist()


def image_to_obb(image: Image) -> Image:
    return Image(
        **{
            **asdict(image),
            "annotations": [
                Annotation(
                    **{
                        **asdict(ann),
                        "bbox": polygon_to_obb(ann.segmentation),
                        "task": "obb",
                    }
                )
                for ann in image.annotations
            ],
        }
    )


def dataset_to_obb(dataset: Dataset) -> Dataset:
    return Dataset(
        images=[image_to_obb(image) for image in dataset.images],
        categories=dataset.categories,
        groups=dataset.groups,
    )
