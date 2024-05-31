from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

import albumentations as A
from shapely.geometry import Polygon

from pflows.polygons import bbox_from_polygon, calculate_center_from_polygon
from pflows.typedef import Annotation, Dataset, Image


def find_biggest_contour(
    contours: Sequence[np.ndarray[np.float32, np.dtype[np.float32]]]
) -> Optional[np.ndarray[np.float32, np.dtype[np.float32]]]:
    max_area: float = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour


def get_new_polygon(
    transformed_mask: Any, polygon_id: int, width: int, height: int
) -> Tuple[float, ...]:
    contours = cv2.findContours(
        (transformed_mask == polygon_id).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    contour = find_biggest_contour(contours)
    if contour is None:
        raise ValueError("No contour found")
    polygon = []
    epsilon = 0.003 * cv2.arcLength(contour, True)
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    try:
        for point in smoothed_contour.squeeze():
            polygon.append(point[0])
            polygon.append(point[1])
    # pylint: disable=broad-exception-caught
    except Exception:
        for point in contour.squeeze():
            polygon.append(point[0])
            polygon.append(point[1])

    return tuple(
        round(p / width, 5) if i % 2 == 0 else round(p / height, 5) for i, p in enumerate(polygon)
    )


def iou_polygons(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    # Convert tuples to numpy arrays
    a_array = np.array(a).reshape(-1, 2)
    b_array = np.array(b).reshape(-1, 2)

    # Create polygon objects
    poly_a = Polygon(a_array)
    poly_b = Polygon(b_array)

    # Calculate intersection and union areas
    intersection_area = float(poly_a.intersection(poly_b).area)
    union_area = float(poly_a.area + poly_b.area - intersection_area)

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def get_new_polygons(
    transformed_mask: Any, polygons: Sequence[Tuple[float, ...]], width: int, height: int
) -> List[Tuple[float, ...]]:
    new_polygons = []
    for polygon_id in np.unique(transformed_mask):
        if polygon_id > 0:
            new_polygon = get_new_polygon(transformed_mask, polygon_id, width, height)
            # Multiply by width and height to get the original polygon
            new_polygon_hw = tuple(
                (
                    int(el * width) if i % 2 == 0 else int(el * height)
                    for i, el in enumerate(new_polygon)
                )
            )
            calculated_iou = iou_polygons(new_polygon_hw, polygons[polygon_id - 1])
            if calculated_iou < 0.3:
                raise ValueError("IoU is too low")
            new_polygons.append(new_polygon)
    return new_polygons


# pylint: disable=too-many-locals
def generate_augmented_image(
    image_id: str,
    image_path: str,
    segments: List[Tuple[float, ...]],
    target_folder: str,
    new_index: int,
) -> Optional[Dict[str, Any]]:
    new_id = f"{image_id}_aug_{new_index}"
    target_file = f"{target_folder}/{new_id}.jpg"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    polygons: List[List[int]] = []
    for polygon_raw in segments:
        polygon = []
        for i in range(0, len(polygon_raw), 2):
            polygon.extend(
                [int(width * float(polygon_raw[i])), int(height * float(polygon_raw[i + 1]))]
            )
        polygons.append(polygon)

    new_polygons = []
    try:
        transform = A.Compose(
            [
                A.Affine(
                    rotate=(-1.5, 1.5),
                    translate_percent=(-0.015, 0.015),
                    scale=(1, 1.2),
                    shear=(-2, 2),
                    p=1.0,
                ),
                A.Defocus(radius=(3, 8), alias_blur=(0.1, 0.4), p=0.7),
                A.CLAHE(clip_limit=2, p=0.9),
                A.RandomBrightnessContrast(p=0.5),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, p=0.2),
                A.GaussNoise(var_limit=(20, 80), mean=50, p=0.8),
                A.ElasticTransform(alpha=1, sigma=25, alpha_affine=25, p=1.0),
            ]
        )

        mask = np.zeros((height, width), dtype=np.uint8)
        for index, polygon_data in enumerate(polygons):
            pts = np.array(
                [polygon_data[i : i + 2] for i in range(0, len(polygon_data), 2)],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], index + 1)  # Use unique IDs as pixel values

        # Apply the transformation
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        new_polygons = get_new_polygons(
            transformed_mask, [tuple(polygon) for polygon in polygons], width, height
        )
        if len(new_polygons) != len(polygons):
            new_polygons = []
            raise ValueError("Different number of polygons")
    # pylint: disable=broad-exception-caught
    except Exception:
        return None

    if len(new_polygons) == 0:
        return None

    cv2.imwrite(target_file, transformed_image)
    return {
        "path": target_file,
        "segments": new_polygons,
        "id": new_id,
        "original_id": image_id,
        "image_index": new_index,
    }


def generate_augmentations(images: List[Image], number: int = 3) -> List[Image]:
    random_temp_folder = f"/tmp/tmp_images_augmented/{uuid4()}"
    os.makedirs(random_temp_folder, exist_ok=True)
    new_images = []
    original_images_by_id = {image.id: image for image in images}

    with ThreadPoolExecutor() as executor:
        futures = []

        for image in images:
            for i in range(number):
                annotation_segments = [
                    annotation.segmentation
                    for annotation in image.annotations
                    if annotation.segmentation is not None
                ]
                futures.append(
                    executor.submit(
                        generate_augmented_image,
                        image.id,
                        image.path,
                        annotation_segments,
                        random_temp_folder,
                        i,
                    )
                )

        for future in as_completed(futures):
            # try:
            new_image_info = future.result()
            if not new_image_info:
                continue
            original_image = original_images_by_id[new_image_info["original_id"]]
            new_images.append(
                Image(
                    id=new_image_info["id"],
                    path=new_image_info["path"],
                    intermediate_ids=original_image.intermediate_ids + [original_image.id],
                    width=original_image.width,
                    height=original_image.height,
                    size_kb=original_image.size_kb,
                    group=original_image.group,
                    annotations=[
                        Annotation(
                            id=f"{annotation.id}_{new_image_info['image_index']}",
                            segmentation=new_segmentation,
                            category_id=annotation.category_id,
                            bbox=bbox_from_polygon(new_segmentation),
                            center=calculate_center_from_polygon(new_segmentation),
                            task=annotation.task,
                            conf=annotation.conf,
                            category_name=annotation.category_name,
                            tags=annotation.tags + ["augmented"],
                        )
                        for annotation, new_segmentation in zip(
                            original_image.annotations, new_image_info["segments"]
                        )
                    ],
                ),
            )
            print("Augmented image", new_images[-1].id)

    return new_images


def generic(dataset: Dataset, number: int = 3) -> Dataset:
    augmented_images = generate_augmentations(dataset.images, number)
    return Dataset(
        images=dataset.images + augmented_images,
        categories=dataset.categories,
        groups=dataset.groups,
    )
