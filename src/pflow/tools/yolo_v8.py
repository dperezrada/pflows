import os
from pathlib import Path
from hashlib import md5
from typing import List, Tuple, Callable, Any

import torch
import yaml
from PIL import Image as ImagePil
from ultralytics import YOLO
import cv2
import numpy as np
from skimage.measure import approximate_polygon
from numpy.typing import NDArray

from pflow.typedef import Annotation, Category, Dataset, Image
from pflow.polygons import (
    calculate_center_from_bbox,
    calculate_center_from_polygon,
    bbox_from_polygon,
    polygon_from_bbox,
)

GROUPS_ALIAS = {"val": "val", "test": "test", "valid": "val", "train": "train"}
ROUNDING = 6


def get_item_from_numpy_or_tensor(element: torch.Tensor | np.ndarray[Any, Any] | Any) -> Any:
    if isinstance(element, torch.Tensor):
        # element is a Tensor
        values = element.numpy().item()
    elif isinstance(element, np.ndarray):
        # element is a NumPy array
        values = element.item()
    else:
        # element is neither a Tensor nor a NumPy array
        values = element
    return values


def get_image_info(image_path: str, group_name: str) -> Image:
    with ImagePil.open(image_path) as img:
        width, height = img.size
        image_bytes = img.tobytes()
        size_bytes = len(image_bytes)
        size_kb = int(round(size_bytes / 1024, 2))
        image_hash = md5(image_bytes).hexdigest()
    image: Image = Image(
        id=image_hash,
        intermediate_ids=[],
        path=str(image_path),
        width=width,
        height=height,
        size_kb=size_kb,
        group=group_name,
    )
    return image


def bbox_from_yolo_v8(
    polygon_row: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x_center, y_center, width, height = polygon_row
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def process_image_annotations(label_path: str, categories: List[Category]) -> List[Annotation]:
    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f.readlines()):
            rows = line.split(" ")
            category_id = int(rows[0])
            polygon_row = [round(float(x), ROUNDING) for x in rows[1:]]
            # Generate a unique id for the annotation
            data = f"{label_path}_{category_id}_{index}_{', '.join(str(x) for x in polygon_row)}"
            md5_hash = md5(data.encode()).hexdigest()
            if len(polygon_row) == 4:
                bbox_row_tuple = (
                    polygon_row[0],
                    polygon_row[1],
                    polygon_row[2],
                    polygon_row[3],
                )
                bbox = bbox_from_yolo_v8(bbox_row_tuple)
                annotations.append(
                    Annotation(
                        id=md5_hash,
                        category_id=category_id,
                        category_name=categories[category_id].name,
                        center=calculate_center_from_bbox(bbox_row_tuple),
                        bbox=bbox,
                        segmentation=polygon_from_bbox(bbox),
                        task="detect",
                    )
                )
            else:
                polygon_row_tuple = tuple(polygon_row)
                annotations.append(
                    Annotation(
                        id=md5_hash,
                        category_id=category_id,
                        category_name=categories[category_id].name,
                        center=calculate_center_from_polygon(polygon_row_tuple),
                        bbox=bbox_from_polygon(polygon_row_tuple),
                        segmentation=polygon_row_tuple,
                        task="segment",
                    )
                )
    return annotations


def load_dataset(folder_path: str) -> Dataset:
    print()
    print("Loading dataset from yolo_v8 format in: ", folder_path)
    data_yaml = Path(folder_path) / "data.yaml"
    if not os.path.exists(data_yaml):
        raise ValueError(f"File {data_yaml} does not exist")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        categories = [Category(name=str(x), id=i) for i, x in enumerate(data["names"])]
        groups = {
            GROUPS_ALIAS[file]: str((Path(folder_path) / file).resolve())
            for file in os.listdir(folder_path)
            if file in GROUPS_ALIAS
        }
        images = []
        for group_name, group_folder in groups.items():
            if not os.path.exists(group_folder):
                raise ValueError(f"Group {group_name} does not exist")

            images_folder = Path(group_folder) / "images"
            for image_path in os.listdir(images_folder):
                image_target_path = images_folder / image_path
                if not os.path.exists(image_target_path):
                    raise ValueError(f"Image {image_target_path} does not exist")
                image_info = get_image_info(str(image_target_path), group_name)
                image_info.annotations = process_image_annotations(
                    str(Path(group_folder) / "labels" / (image_target_path.stem + ".txt")),
                    categories,
                )
                images.append(image_info)
        return Dataset(
            images=images,
            categories=categories,
        )


def preprocess_image(
    image: ImagePil.Image,
    new_size: Tuple[int, int] = (640, 640),
    grayscale: bool = False,
) -> NDArray[np.uint8]:
    # 1. Stretch to 640x640
    new_image = image.resize(new_size)

    # Convert to numpy array for OpenCV processing
    image_np = np.array(new_image)

    if grayscale:
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            # already in grayscale
            pass
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_np = clahe.apply(image_np)
    else:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image_np.astype(np.uint8)


# work in the pylint errors
# pylint: disable=too-many-arguments,too-many-locals
def run_model_on_image(
    image_path: str,
    model: YOLO,
    threshold: float = 0.5,
    segment_tolerance: float = 0.02,
    preprocess_function: Callable[[ImagePil.Image], NDArray[np.uint8]] = preprocess_image,
    add_tag: str | None = None,
) -> List[Annotation]:
    # We run the model on the image
    image = ImagePil.open(image_path)

    results = model.predict(
        preprocess_function(image),
        conf=threshold,
    )
    annotations = []
    model_categories = model.names
    if isinstance(model_categories, dict):
        model_categories = [model_categories[key] for key in sorted(model_categories.keys())]
    for result in results:
        # Segmentation case
        if result.masks is not None and result.boxes is not None:
            for np_mask, box in zip(result.masks.xyn, result.boxes):
                category_id = int(box.cls[0])
                category_name = model_categories[category_id]

                points = np_mask.ravel().tolist()
                polygon_array = np.array(points).reshape(-1, 2)
                simplified_polygon = approximate_polygon(
                    polygon_array, tolerance=segment_tolerance
                )  # type: ignore[no-untyped-call]
                simplified_points = tuple(
                    round(x, ROUNDING) for x in simplified_polygon.flatten().tolist()
                )

                joined_points = ", ".join(str(x) for x in simplified_points)
                hash_id = md5(
                    f"{image_path}_{category_id}_{box.conf[0]}_{joined_points}".encode()
                ).hexdigest()
                annotations.append(
                    Annotation(
                        id=hash_id,
                        category_id=category_id,
                        category_name=category_name,
                        center=calculate_center_from_polygon(simplified_points),
                        bbox=bbox_from_polygon(simplified_points),
                        segmentation=simplified_points,
                        task="segment",
                        conf=round(get_item_from_numpy_or_tensor(box.conf[0]), ROUNDING),
                        tags=[add_tag] if add_tag else [],
                    )
                )
            continue
        # Classification case
        if result.boxes is None and result.probs is not None:
            for index, prob in enumerate(result.probs.numpy().tolist()):
                # we get the index of the class with the highest probability
                if prob < threshold:
                    continue
                category_name = model_categories[index]
                annotations.append(
                    Annotation(
                        id=md5(f"{image_path}_{index}_{prob}".encode()).hexdigest(),
                        category_id=index,
                        category_name=category_name,
                        center=None,
                        bbox=None,
                        segmentation=None,
                        task="classification",
                        conf=round(prob, ROUNDING),
                        tags=[add_tag] if add_tag else [],
                    )
                )
            continue

        # Bounding box case
        if result.boxes is not None:
            for box in result.boxes:
                bbox = tuple(round(x, ROUNDING) for x in box.xyxyn[0].tolist())
                category_id = int(get_item_from_numpy_or_tensor(box.cls))
                category_name = model_categories[category_id]
                confidence = round(get_item_from_numpy_or_tensor(box.conf), ROUNDING)
                hash_id = md5(
                    (
                        f"{image_path}_{category_id}_{confidence}_{', '.join(str(x) for x in bbox)}"
                    ).encode()
                ).hexdigest()
                annotations.append(
                    Annotation(
                        id=hash_id,
                        category_id=category_id,
                        category_name=category_name,
                        center=calculate_center_from_bbox(bbox),
                        bbox=bbox,
                        segmentation=polygon_from_bbox(bbox),
                        task="detect",
                        conf=confidence,
                        tags=[add_tag] if add_tag else [],
                    )
                )
    return sorted(
        annotations,
        key=lambda x: x.conf,
        reverse=True,
    )


def run_model(
    dataset: Dataset,
    model_path: str,
    add_tag: str | None = None,
    threshold: float = 0.5,
    segment_tolerance: float = 0.02,
) -> Dataset:
    model = YOLO(model_path)
    for image in dataset.images:
        model_annotations = run_model_on_image(
            image.path,
            model,
            threshold=threshold,
            segment_tolerance=segment_tolerance,
            add_tag=add_tag,
        )
        image.annotations = image.annotations + model_annotations
    return dataset


def write(dataset: Dataset, target_dir: str) -> None:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    data_yaml = target_path / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "names": [category.name for category in dataset.categories],
                "nc": len(dataset.categories),
            },
            f,
        )
    total_images: dict[str, int] = {}
    for image in dataset.images:
        image_folder = target_path / image.group / "images"
        image_folder.mkdir(parents=True, exist_ok=True)
        image_path = image_folder / f"{image.id}.jpg"
        with ImagePil.open(image.path) as img:
            img.save(image_path)
        total_images[image.group] = total_images.get(image.group, 0) + 1
        label_folder = target_path / image.group / "labels"
        label_folder.mkdir(parents=True, exist_ok=True)
        label_path = label_folder / f"{image.id}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for annotation in image.annotations:
                if annotation.task == "detect" and annotation.bbox:
                    f.write(
                        f"{annotation.category_id} {' '.join(str(x) for x in annotation.bbox)}\n"
                    )
                elif annotation.task == "segment" and annotation.segmentation:
                    polygon_str = " ".join(str(x) for x in annotation.segmentation)
                    f.write(f"{annotation.category_id} {polygon_str}\n")
    print()
    print("Dataset saved as yolo_v8 format in: ", target_dir)
    print("total images: ", len(dataset.images))
    print("total images per group:")
    for group, total in total_images.items():
        print(f"\t{group}: {total}")
    print()
