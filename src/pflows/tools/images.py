import math
import tempfile
import colorsys
from dataclasses import asdict
from typing import List
from uuid import uuid4

# from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import cv2
from PIL import Image as ImagePil
from PIL import ImageDraw
from shapely.geometry import Polygon
from shapely.ops import transform
import numpy as np

from pflows.model import get_image_info
from pflows.polygons import (
    bbox_from_polygon,
    calculate_center_from_bbox,
    calculate_center_from_polygon,
)
from pflows.typedef import Annotation, Dataset, Image


# TODO: refactor this function
# pylint: disable=too-many-locals
def crop_to_annotations(dataset: Dataset, tag=None) -> Dataset:
    """
    Crop images to match the min and max x/y of the annotations.
    """
    temp_images_path = tempfile.mkdtemp()
    new_images: List[Image] = []
    for image in dataset.images:
        pil_image = ImagePil.open(image.path)
        width, height = pil_image.size
        min_x = 1000000
        min_y = 1000000
        max_x = 0
        max_y = 0

        for annotation in image.annotations:
            if annotation.bbox is None or (tag is not None and tag not in annotation.tags):
                continue
            min_x = min(int(min(annotation.bbox[0::2]) * width), min_x)
            min_y = min(int(min(annotation.bbox[1::2]) * height), min_y)
            max_x = max(int(max(annotation.bbox[0::2]) * width), max_x)
            max_y = max(int(max(annotation.bbox[1::2]) * height), max_y)
        new_width = int((max_x - min_x))
        new_height = int((max_y - min_y))
        # crop the image
        # Adjust annotations for the cropped image
        for annotation in image.annotations:
            if annotation.bbox is not None and (tag is None or tag in annotation.tags):
                (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = tuple(
                    (
                        (width * coord - min_x) / new_width
                        if i % 2 == 0
                        else (height * coord - min_y) / new_height
                    )
                    for i, coord in enumerate(annotation.bbox)
                )
                annotation.bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

            if annotation.center is not None:
                (center_x, center_y) = tuple(
                    (
                        (width * coord - min_x) / new_width
                        if i % 2 == 0
                        else (height * coord - min_y) / new_height
                    )
                    for i, coord in enumerate(annotation.center)
                )
                annotation.center = (center_x, center_y)

            if annotation.segmentation is not None:
                new_segmentation = tuple(
                    (
                        (width * coord - min_x) / new_width
                        if i % 2 == 0
                        else (height * coord - min_y) / new_height
                    )
                    for i, coord in enumerate(annotation.segmentation)
                )
                annotation.segmentation = new_segmentation

        cropped_pil_image = pil_image.crop(
            (
                min_x,
                min_y,
                max_x,
                max_y,
            )
        )
        new_path = f"{temp_images_path}/{image.id}.jpg"
        cropped_pil_image.save(new_path)
        new_image = get_image_info(new_path, image.group, image.intermediate_ids + [image.id])
        new_images.append(
            Image(**{**asdict(image), **asdict(new_image), "annotations": image.annotations})
        )
    return Dataset(images=new_images, categories=dataset.categories, groups=dataset.groups)


def create_mask(annotation: Annotation, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    if annotation.segmentation:
        annotations_absolute = [
            value * width if i % 2 == 0 else value * height
            for i, value in enumerate(annotation.segmentation)
        ]
        points = np.array(annotations_absolute, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [points], 1)
    return mask


def crop_mask(mask, crop_x, crop_y, crop_width, crop_height):
    # We need to count the number of mask elements left behind
    # when we crop the mask
    mask_elements = np.sum(mask[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width])
    original_elements = np.sum(mask)
    difference = original_elements - mask_elements
    return mask[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width], bool(difference > 0)


def generate_polygons_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [Polygon(cnt.reshape(-1, 2)) for cnt in contours if len(cnt) >= 3]
    return polygons


def adjust_polygon_coords(polygon, crop_width, crop_height):
    def adjust_coords(x, y):
        return x / crop_width, y / crop_height

    return transform(adjust_coords, polygon)


def make_id() -> str:
    return str(uuid4())


def get_segmentation_annotations_from_sliced_image(
    sliced_image: Image, original_image: Image, crop_x: int, crop_y: int
) -> List[Annotation]:
    sliced_annotations = []
    w, h = sliced_image.width, sliced_image.height

    for annotation in original_image.annotations or []:
        mask = create_mask(annotation, original_image.width, original_image.height)
        cropped_mask, truncated = crop_mask(mask, crop_x, crop_y, w, h)
        polygons = generate_polygons_from_mask(cropped_mask)

        for polygon in polygons:
            adjusted_polygon = adjust_polygon_coords(polygon, w, h)

            adjusted_segment = tuple(
                float(coord) for point in adjusted_polygon.exterior.coords for coord in point
            )

            sliced_annotation = Annotation(
                id=make_id(),
                category_id=annotation.category_id,
                original_id=annotation.id,
                center=calculate_center_from_polygon(adjusted_segment),
                bbox=annotation.bbox,
                segmentation=tuple(adjusted_segment),
                task=annotation.task,
                conf=annotation.conf,
                category_name=annotation.category_name,
                tags=annotation.tags,
                truncated=truncated,
            )
            sliced_annotations.append(sliced_annotation)

    return sliced_annotations


def get_bbox_annotations_from_sliced_image(
    sliced_image: Image, original_image: Image, crop_x: int, crop_y: int
) -> List[Annotation]:
    sliced_annotations = []
    w, h = sliced_image.width, sliced_image.height

    for annotation in original_image.annotations:
        if annotation.bbox is not None:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = annotation.bbox
            # Convert to absolute coordinates
            bbox_x1 = bbox_x1 * original_image.width
            bbox_y1 = bbox_y1 * original_image.height
            bbox_x2 = bbox_x2 * original_image.width
            bbox_y2 = bbox_y2 * original_image.height

            # Check if the bounding box is completely outside the crop
            if (
                bbox_x2 <= crop_x
                or bbox_x1 >= crop_x + w
                or bbox_y2 <= crop_y
                or bbox_y1 >= crop_y + h
            ):
                continue

            # Calculate the intersection of the bounding box with the sliced image
            intersection_x1 = max(bbox_x1, crop_x)
            intersection_y1 = max(bbox_y1, crop_y)
            intersection_x2 = min(bbox_x2, crop_x + w)
            intersection_y2 = min(bbox_y2, crop_y + h)

            # Calculate the adjusted bounding box coordinates
            adjusted_bbox_x1 = max(0, intersection_x1 - crop_x)
            adjusted_bbox_y1 = max(0, intersection_y1 - crop_y)
            adjusted_bbox_x2 = min(w, intersection_x2 - crop_x)
            adjusted_bbox_y2 = min(h, intersection_y2 - crop_y)

            truncated_bbox = False
            if (
                adjusted_bbox_x1 == 0
                or adjusted_bbox_y1 == 0
                or adjusted_bbox_x2 == w
                or adjusted_bbox_y2 == h
            ):
                truncated_bbox = True

            adjusted_bbox_x1 /= w
            adjusted_bbox_y1 /= h
            adjusted_bbox_x2 /= w
            adjusted_bbox_y2 /= h

            sliced_annotation = Annotation(
                id=make_id(),
                category_id=annotation.category_id,
                center=calculate_center_from_bbox(
                    (adjusted_bbox_x1, adjusted_bbox_y1, adjusted_bbox_x2, adjusted_bbox_y2)
                ),
                bbox=(adjusted_bbox_x1, adjusted_bbox_y1, adjusted_bbox_x2, adjusted_bbox_y2),
                segmentation=annotation.segmentation,
                task=annotation.task,
                conf=annotation.conf,
                category_name=annotation.category_name,
                tags=annotation.tags,
                truncated=truncated_bbox,
                original_id=annotation.id,
            )
            sliced_annotations.append(sliced_annotation)

    return sliced_annotations


# pylint: disable=too-many-arguments
def slice_one_image(
    image: Image,
    slice_height: int = 640,
    slice_width: int = 640,
    min_overlap_height_ratio: float = 0.1,
    min_overlap_width_ratio: float = 0.1,
    keep_original_images: bool = False,
) -> List[Image]:

    temp_folder = tempfile.mkdtemp()

    try:
        task_type = image.annotations[0].task
    except:
        task_type = "detect"
    # Only detect or segment is supported for now
    if not task_type in ["detect", "segment", "obb"]:
        return [image]

    sliced_images = []
    img = ImagePil.open(image.path)

    # Calculate the number of slices in each dimension
    num_slices_height = math.ceil(image.height / (slice_height * (1 - min_overlap_height_ratio)))
    num_slices_width = math.ceil(image.width / (slice_width * (1 - min_overlap_width_ratio)))

    if num_slices_height == 1 and num_slices_width == 1:
        if keep_original_images:
            return [image]
        return []
    if num_slices_height == 1:
        num_slices_height = 2
    if num_slices_width == 1:
        num_slices_width = 2
    # Calculate the overlap in pixels
    overlap_height = (num_slices_height * slice_height - image.height) // (num_slices_height - 1)
    overlap_width = (num_slices_width * slice_width - image.width) // (num_slices_width - 1)

    for i in range(num_slices_height):
        for j in range(num_slices_width):
            # Calculate the coordinates for cropping
            left = j * (slice_width - overlap_width)
            top = i * (slice_height - overlap_height)
            right = min(left + slice_width, image.width)
            bottom = min(top + slice_height, image.height)

            sliced_img = img.crop((left, top, right, bottom))
            sliced_path = f"{temp_folder}/slice_{image.id}_{left}_{top}_{right}_{bottom}.jpg"
            if sliced_img.mode == "RGBA":
                sliced_img = sliced_img.convert("RGB")
            sliced_img.save(sliced_path)
            sliced_image = Image(
                id=f"{image.id}_{len(sliced_images)}",
                path=sliced_path,
                intermediate_ids=image.intermediate_ids + [image.id],
                width=slice_width,
                height=slice_height,
                size_kb=sliced_img.size[0] * sliced_img.size[1] // 1024,
                group=image.group,
                tags=image.tags + ["slice"],
                info={
                    "sliced": {
                        "original_path": image.path,
                        "original_width": image.width,
                        "original_height": image.height,
                        "original_image_id": image.id,
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    }
                },
            )
            if task_type in ["segment", "obb"]:
                annotations = get_segmentation_annotations_from_sliced_image(
                    sliced_image, image, left, top
                )
            else:
                annotations = get_bbox_annotations_from_sliced_image(sliced_image, image, left, top)
            sliced_image.annotations = annotations
            sliced_images.append(sliced_image)
    return sliced_images


# def select_best_images_from_slices(sliced_images: List[Image]) -> List[Image]:
#     # Step 1: Count the number of non-truncated annotations for each image
#     image_scores = []
#     images_by_id = {image.id: image for image in sliced_images}
#     for image in sliced_images:
#         non_truncated_count = sum(1 for annotation in image.annotations if not annotation.truncated)
#         image_scores.append((image.id, non_truncated_count))

#     # Step 2: Sort images by the number of non-truncated annotations in descending order
#     image_scores.sort(key=lambda x: x[1], reverse=True)

#     # Step 3: Select images, ensuring all annotations are covered at least once
#     selected_images = set()
#     covered_annotations = set()

#     # Function to get annotations for an image
#     def get_annotations(image_id):
#         for image in sliced_images:
#             if image.id == image_id:
#                 return image.annotations
#         return []

#     all_annotations = {
#         annotation.original_id for image in sliced_images for annotation in image.annotations
#     }

#     for image_id, _ in image_scores:
#         annotations = get_annotations(image_id)
#         current_coverage = {
#             annotation.original_id for annotation in annotations if not annotation.truncated
#         }
#         if not current_coverage.issubset(covered_annotations):
#             selected_images.add(image_id)
#             covered_annotations.update(current_coverage)
#         if covered_annotations == all_annotations:
#             break

#     # Output the selected images
#     return [images_by_id[image_id] for image_id in selected_images]


def is_truncated(annotation: Annotation, image: Image, border_tolerance: int = 10) -> bool:
    if annotation.bbox:
        x1, y1, x2, y2 = annotation.bbox
        return (
            x1 * image.width < border_tolerance
            or y1 * image.height < border_tolerance
            or x2 * image.width > image.width - border_tolerance
            or y2 * image.height > image.height - border_tolerance
        )
    elif annotation.segmentation:
        points = annotation.segmentation
        for i in range(0, len(points), 2):
            x, y = points[i] * image.width, points[i + 1] * image.height
            if (
                x < border_tolerance
                or y < border_tolerance
                or x > image.width - border_tolerance
                or y > image.height - border_tolerance
            ):
                return True
        return False
    return False


def select_best_images_from_slices(
    sliced_images: List[Image], border_tolerance: int = 10
) -> List[Image]:
    # Step 1: Count the number of non-truncated annotations for each image
    image_scores = []
    images_by_id = {image.id: image for image in sliced_images}
    for image in sliced_images:
        non_truncated_count = sum(
            1
            for annotation in image.annotations
            if not is_truncated(annotation, image, border_tolerance)
        )
        image_scores.append((image.id, non_truncated_count))

    # Step 2: Sort images by the number of non-truncated annotations in descending order
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Select images, ensuring all annotations are covered at least once
    selected_images = set()
    covered_annotations = set()

    # Function to get annotations for an image
    def get_annotations(image_id):
        return images_by_id[image_id].annotations

    all_annotations = {
        annotation.original_id for image in sliced_images for annotation in image.annotations
    }

    for image_id, _ in image_scores:
        annotations = get_annotations(image_id)
        current_coverage = {
            annotation.original_id
            for annotation in annotations
            if not is_truncated(annotation, images_by_id[image_id], border_tolerance)
        }
        if not current_coverage.issubset(covered_annotations):
            selected_images.add(image_id)
            covered_annotations.update(current_coverage)
        if covered_annotations == all_annotations:
            break

    # Output the selected images
    return [images_by_id[image_id] for image_id in selected_images]


# pylint: disable=too-many-statements
def join_slices(sliced_images: List[Image]) -> List[Image]:
    # Step 1: Group slices by original image
    slices_by_original_image = {}
    for sliced_image in sliced_images:
        original_image_id = sliced_image.info["sliced"]["original_image_id"]
        if original_image_id not in slices_by_original_image:
            slices_by_original_image[original_image_id] = []
        slices_by_original_image[original_image_id].append(sliced_image)

    # Step 2: Create a new image for each original image
    joined_images = []
    for original_image_id, slices in slices_by_original_image.items():
        first_slice = slices[0]
        original_image_path = first_slice.info["sliced"]["original_path"]
        original_image_width = first_slice.info["sliced"]["original_width"]
        original_image_height = first_slice.info["sliced"]["original_height"]
        # original_image_annotations = original_image.annotations

        joined_image = get_image_info(
            original_image_path,
            first_slice.group,
            [original_image_id] + [slice.id for slice in slices],
        )
        joined_image.tags = first_slice.tags
        joined_image.annotations = []

        already_added = set()
        # Step 3: We need to adjust the annotations for the joined image
        # using the new width and height
        for slice_image in slices:
            for annotation in slice_image.annotations:
                if annotation.id in already_added:
                    continue
                if annotation.truncated:
                    continue
                if annotation.segmentation:
                    segmentation = list(annotation.segmentation)
                    for i in range(0, len(segmentation), 2):
                        x = segmentation[i] * slice_image.width
                        y = segmentation[i + 1] * slice_image.height

                        x += slice_image.info["sliced"]["left"]
                        y += slice_image.info["sliced"]["top"]

                        x /= original_image_width
                        y /= original_image_height

                        segmentation[i] = x
                        segmentation[i + 1] = y

                    tuple_segmentation = tuple(segmentation)
                    annotation.segmentation = tuple_segmentation
                    annotation.center = calculate_center_from_polygon(tuple_segmentation)
                    annotation.bbox = bbox_from_polygon(tuple_segmentation)
                    already_added.add(annotation.id)
                    joined_image.annotations.append(annotation)
                    continue
                if annotation.bbox:
                    x1, y1, x2, y2 = annotation.bbox
                    x1 = x1 * slice_image.width
                    y1 = y1 * slice_image.height
                    x2 = x2 * slice_image.width
                    y2 = y2 * slice_image.height

                    x1 += slice_image.info["sliced"]["left"]
                    y1 += slice_image.info["sliced"]["top"]
                    x2 += slice_image.info["sliced"]["left"]
                    y2 += slice_image.info["sliced"]["top"]

                    x1 /= original_image_width
                    y1 /= original_image_height
                    x2 /= original_image_width
                    y2 /= original_image_height

                    annotation.bbox = (x1, y1, x2, y2)
                    annotation.center = calculate_center_from_bbox(annotation.bbox)
                    joined_image.annotations.append(annotation)
                    already_added.add(annotation.id)
                    continue

        joined_images.append(joined_image)
    return joined_images


def slice_dataset(
    dataset: Dataset,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.1,
    keep_original_images: bool = False,
    select_best: bool = True,
    border_tolerance: int = 10,
) -> Dataset:
    sliced_images = []
    for image in dataset.images:
        sliced_images.extend(
            slice_one_image(
                image,
                slice_height,
                slice_width,
                overlap_height_ratio,
                overlap_width_ratio,
                keep_original_images,
            )
        )
    if select_best:
        sliced_images = select_best_images_from_slices(sliced_images, border_tolerance)
    return Dataset(images=sliced_images, categories=dataset.categories, groups=dataset.groups)


def join_slice_dataset(
    dataset: Dataset,
) -> Dataset:
    joined_images = join_slices(dataset.images)
    return Dataset(images=joined_images, categories=dataset.categories, groups=dataset.groups)


def write_image_with_annotations(image: Image, target_image_path: str | None = None):
    img = ImagePil.open(image.path)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Generate a color map for different class_ids
    unique_class_ids = set(ann.category_id for ann in image.annotations)
    color_map = {
        class_id: tuple(
            int(x * 255) for x in colorsys.hsv_to_rgb(i / len(unique_class_ids), 1.0, 1.0)
        )
        for i, class_id in enumerate(unique_class_ids)
    }

    for annotation in image.annotations:
        color = color_map[annotation.category_id]
        if annotation.segmentation:
            segmentation = annotation.segmentation
            points = [
                (segmentation[i] * image.width, segmentation[i + 1] * image.height)
                for i in range(0, len(segmentation), 2)
            ]
            draw.polygon(points, outline=color)
        elif annotation.bbox:
            x1, y1, x2, y2 = annotation.bbox
            x1 = x1 * image.width
            y1 = y1 * image.height
            x2 = x2 * image.width
            y2 = y2 * image.height
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    if target_image_path:
        img.save(target_image_path)
    return img


def remove_all_annotations(dataset: Dataset) -> Dataset:
    return Dataset(
        images=[Image(**{**asdict(image), "annotations": []}) for image in dataset.images],
        categories=dataset.categories,
        groups=dataset.groups,
    )
