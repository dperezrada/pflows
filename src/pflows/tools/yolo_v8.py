import os
import json
from datetime import datetime, date
from pathlib import Path
from hashlib import md5
import shutil
import tempfile
from typing import List, Tuple, Callable, Any, Dict
import zipfile
from multiprocessing import Pool, cpu_count

import torch
import yaml
from PIL import Image as ImagePil
from ultralytics import YOLO
import cv2
import numpy as np
from skimage.measure import approximate_polygon
from numpy.typing import NDArray

from pflows.tools.categories import remap_category_ids
from pflows.typedef import Annotation, Category, Dataset
from pflows.polygons import (
    calculate_center_from_bbox,
    calculate_center_from_polygon,
    bbox_from_polygon,
    polygon_from_bbox,
)
from pflows.model import get_image_info

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


def bbox_from_yolo_v8(
    polygon_row: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x_center, y_center, width, height = polygon_row
    return (
        round(x_center - width / 2, ROUNDING),
        round(y_center - height / 2, ROUNDING),
        round(x_center + width / 2, ROUNDING),
        round(y_center + height / 2, ROUNDING),
    )


def yolov8_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return x_center, y_center, width, height


def process_image_annotations(
    label_path: str, categories: List[Category], mode: str = "auto"
) -> List[Annotation]:
    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f.readlines()):
            rows = line.split(" ")
            category_id = int(rows[0])
            polygon_row = [round(float(x), ROUNDING) for x in rows[1:]]
            # Generate a unique id for the annotation
            data = f"{label_path}_{category_id}_{index}_{', '.join(str(x) for x in polygon_row)}"
            md5_hash = md5(data.encode()).hexdigest()
            bbox = None
            polygon = None
            obb = None
            candidate_task = None
            if len(polygon_row) == 4:
                bbox_row_tuple = (
                    polygon_row[0],
                    polygon_row[1],
                    polygon_row[2],
                    polygon_row[3],
                )
                bbox = bbox_from_yolo_v8(bbox_row_tuple)
                polygon = polygon_from_bbox(bbox)
                task = "detect"
            else:
                if len(polygon_row) == 8:
                    task = "obb"
                    obb = (
                        polygon_row[0],
                        polygon_row[1],
                        polygon_row[2],
                        polygon_row[3],
                        polygon_row[4],
                        polygon_row[5],
                        polygon_row[6],
                        polygon_row[7],
                    )
                else:
                    task = "segment"
                polygon = tuple(polygon_row)
                bbox = bbox_from_polygon(polygon)
            if mode == "bbox":
                task = "detect"
            elif mode == "segment":
                task = "segment"

            annotations.append(
                Annotation(
                    id=md5_hash,
                    category_id=category_id,
                    category_name=categories[category_id].name,
                    center=calculate_center_from_polygon(polygon),
                    bbox=bbox,
                    segmentation=polygon,
                    task=task,
                    obb=obb,
                )
            )
    return annotations


def get_categories_from_model(model):
    categories = []
    if isinstance(model.names, dict):
        for key in get_model_category_ids(model):
            categories.append(model.names[key])
    else:
        categories = model.names
    return categories


def get_model_category_ids(model):
    model_names_keys = sorted(
        model.names.keys() if isinstance(model.names, dict) else range(len(model.names))
    )
    return model_names_keys


def load_categories(parsed_yaml_file) -> List[Category]:
    classes = parsed_yaml_file["names"]
    if isinstance(classes, dict):
        classes = list(classes.values())
    return [Category(name=str(cls), id=index) for index, cls in enumerate(classes)]


def load_dataset(
    dataset: Dataset | None, folder_path: str, number: int | None = None, mode: str = "auto"
) -> Dataset:
    if dataset is None:
        dataset = Dataset(images=[], categories=[], groups=[])
    print()
    print("Loading dataset from yolo_v8 format in: ", folder_path)
    data_yaml = Path(folder_path) / "data.yaml"
    if not os.path.exists(data_yaml):
        raise ValueError(f"File {data_yaml} does not exist")
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        categories = load_categories(data)
        groups = {
            GROUPS_ALIAS[file]: str((Path(folder_path) / file).resolve())
            for file in os.listdir(folder_path)
            if file in GROUPS_ALIAS
        }
        images = []
        reach_max = False
        for group_name, group_folder in groups.items():
            if reach_max:
                break
            if not os.path.exists(group_folder):
                raise ValueError(f"Group {group_name} does not exist")

            images_folder = Path(group_folder) / "images"
            for image_path in os.listdir(images_folder):
                if number is not None and len(images) >= number:
                    reach_max = True
                    break
                image_target_path = images_folder / image_path
                if not os.path.exists(image_target_path):
                    raise ValueError(f"Image {image_target_path} does not exist")
                image_info = get_image_info(str(image_target_path), group_name)
                if not os.path.exists(image_info.path):
                    image_info.path = str(image_target_path)
                image_info.annotations = process_image_annotations(
                    str(Path(group_folder) / "labels" / (image_target_path.stem + ".txt")),
                    categories,
                    mode,
                )
                images.append(image_info)
        category_names = list(set([category.name for category in categories + dataset.categories]))
        new_groups = list(set(list(groups.keys()) + dataset.groups))
        return remap_category_ids(
            Dataset(
                images=images + dataset.images,
                categories=[
                    Category(name=name, id=index) for index, name in enumerate(category_names)
                ],
                groups=new_groups,
            )
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

    # Check the number of channels
    if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
        # Image is grayscale
        if not grayscale:
            # Convert to RGB if color output is desired
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3:
        if image_np.shape[2] == 4:
            # Image is RGBA, convert to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 3:
            # Image is RGB, no conversion needed
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")

        if grayscale:
            # Convert to grayscale if grayscale output is desired
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image_np.shape}")

    if grayscale:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_np = clahe.apply(image_np)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
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
    model_categories: List[str] | None = None,
    threshold: float = 0.5,
    segment_tolerance: float = 0,
    preprocess_function: Callable[[ImagePil.Image], NDArray[np.uint8]] = preprocess_image,
    add_tag: str | None = None,
    model_id: str | None = None,
) -> List[Annotation]:
    # We run the model on the image
    image = ImagePil.open(image_path)

    results = model.predict(
        preprocess_function(image),
        conf=threshold,
    )
    annotations = []
    yolo_categories = model.names
    if isinstance(yolo_categories, dict):
        yolo_categories = [str(yolo_categories[key]) for key in sorted(yolo_categories.keys())]
    if model_categories is None:
        model_categories = yolo_categories
    task = model.task
    print(task)
    if not task in ["detect", "segment", "obb"]:
        raise ValueError(f"Task {task} is not supported")
    task = str(task)
    for result in results:
        # Segmentation case
        if result.masks is not None and result.boxes is not None:
            for np_mask, box in zip(result.masks.xyn, result.boxes):
                category_id = int(box.cls[0])
                category_name = model_categories[category_id]

                points = np_mask.ravel().tolist()
                polygon_array = np.array(points).reshape(-1, 2)
                # We didn't got good results with the approximate_polygon
                # simplified_polygon = approximate_polygon(
                #     polygon_array, tolerance=segment_tolerance
                # )  # type: ignore[no-untyped-call]
                # simplified_points = tuple(
                #     round(x, ROUNDING) for x in simplified_polygon.flatten().tolist()
                # )
                simplified_points = tuple(
                    round(x, ROUNDING) for x in polygon_array.flatten().tolist()
                )
                if len(simplified_points) < 4:
                    continue

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
                        task=task,
                        conf=round(get_item_from_numpy_or_tensor(box.conf[0]), ROUNDING),
                        tags=[add_tag] if add_tag else [],
                        model_id=model_id,
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
                        model_id=model_id,
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
                        model_id=model_id,
                    )
                )
        if result.obb is not None:
            for obb in result.obb:
                category_id = int(get_item_from_numpy_or_tensor(obb.cls))
                category_name = model_categories[category_id]
                confidence = round(get_item_from_numpy_or_tensor(obb.conf), ROUNDING)

                # Extract the OBB normalized coordinates
                obb_coords = obb.xyxyxyxyn.tolist()[0]  # Normalized coordinates

                # Calculate the center of the OBB (normalized)
                center_x = sum(x for x, _ in obb_coords) / 4
                center_y = sum(y for _, y in obb_coords) / 4
                center = (round(center_x, ROUNDING), round(center_y, ROUNDING))

                # Calculate an approximate bounding box (normalized)
                x_coords, y_coords = zip(*obb_coords)
                bbox = (
                    round(min(x_coords), ROUNDING),
                    round(min(y_coords), ROUNDING),
                    round(max(x_coords), ROUNDING),
                    round(max(y_coords), ROUNDING),
                )

                # Round the segmentation coordinates
                segmentation_pairs = [
                    (round(x, ROUNDING), round(y, ROUNDING)) for x, y in obb_coords
                ]
                segmentation_flat = [item for sublist in segmentation_pairs for item in sublist]

                hash_id = md5(
                    f"{image_path}_{category_id}_{confidence}_{','.join(map(str, segmentation_flat))}".encode()
                ).hexdigest()

                annotations.append(
                    Annotation(
                        id=hash_id,
                        category_id=category_id,
                        category_name=category_name,
                        center=center,
                        bbox=bbox,
                        segmentation=segmentation_flat,
                        obb=segmentation_flat,
                        task="obb",
                        conf=confidence,
                        tags=[add_tag] if add_tag else [],
                        model_id=model_id,
                    )
                )
    return sorted(
        annotations,
        key=lambda x: x.conf,
        reverse=True,
    )


def performance_non_max_suppression(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression (NMS) on the bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes, each represented as [x1, y1, x2, y2].
        scores (numpy.ndarray): Array of corresponding confidence scores for each box.
        iou_threshold (float): Intersection over Union (IoU) threshold for
        suppressing overlapping boxes.

    Returns:
        list: Indices of the selected boxes after applying NMS.
    """
    # Sort the boxes by their confidence scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    selected_indices = []

    while len(sorted_indices) > 0:
        # Select the box with the highest confidence score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Compute the IoU between the current box and the remaining boxes
        current_box = boxes[current_index]
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        # Compute the coordinates of the intersection area
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        # Compute the area of the intersection
        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute the area of the current box and the remaining boxes
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
            remaining_boxes[:, 3] - remaining_boxes[:, 1]
        )

        # Compute the IoU
        iou = intersection_area / (current_area + remaining_areas - intersection_area)

        # Remove the boxes with IoU greater than the threshold
        keep_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = remaining_indices[keep_indices]

    return selected_indices


def annotations_non_max_suppression(
    annotations: List[Annotation],
    categories=None,
    iou_threshold: float = 0.4,
    active: bool = True,
) -> List[Annotation]:
    keep_annotations = []
    if categories is None:
        categories = [annotation.category_id for annotation in annotations]
    for category_id in categories:
        category_annotations = [
            annotation for annotation in annotations if annotation.category_id == category_id
        ]
        if not category_annotations:
            continue
        if active:
            keep_annotations_indexes = performance_non_max_suppression(
                np.array([annotation.bbox for annotation in category_annotations]),
                np.array([annotation.conf for annotation in category_annotations]),
                iou_threshold,
            )
            keep_annotations.extend(
                [category_annotations[index] for index in keep_annotations_indexes]
            )
        else:
            keep_annotations.extend(category_annotations)
    return keep_annotations


def process_image(args):
    (
        image,
        model_path,
        threshold,
        segment_tolerance,
        add_tag,
        non_max_suppression,
        non_max_suppression_threshold,
        model_names_keys,
    ) = args
    model = YOLO(model_path)
    model_annotations = run_model_on_image(
        image.path,
        model,
        threshold=threshold,
        segment_tolerance=segment_tolerance,
        add_tag=add_tag,
    )
    keep_annotations = annotations_non_max_suppression(
        model_annotations,
        model_names_keys,
        iou_threshold=non_max_suppression_threshold,
        active=non_max_suppression,
    )
    return keep_annotations


def run_model(
    dataset: Dataset,
    model_path: str | YOLO,
    add_tag: str | None = None,
    threshold: float = 0.5,
    segment_tolerance: float = 0.02,
    non_max_suppression: bool = True,
    non_max_suppression_threshold: float = 0.4,
    parallel: bool = False,
    num_processes: int | None = None,
) -> Dataset:
    if isinstance(model_path, str):
        model = YOLO(model_path)
    else:
        model = model_path
    model_names_keys = get_model_category_ids(model)
    model_names = [model.names[key] for key in sorted(model_names_keys)]
    current_category_names = [category.name for category in dataset.categories]
    new_categories = current_category_names + [
        name for name in model_names if name not in current_category_names
    ]
    print("New categories found: ", new_categories)
    total_images = len(dataset.images)

    if parallel:
        # Prepare arguments for parallel processing
        args_list = [
            (
                image,
                model_path,
                threshold,
                segment_tolerance,
                add_tag,
                non_max_suppression,
                non_max_suppression_threshold,
                model_names_keys,
            )
            for image in dataset.images
        ]

        if num_processes is None:
            num_processes = cpu_count() - 1

        # Create process pool and process images in parallel
        with Pool(processes=num_processes) as pool:
            results = []
            for i, keep_annotations in enumerate(pool.imap_unordered(process_image, args_list)):
                dataset.images[i].annotations = dataset.images[i].annotations + keep_annotations
                if i % 25 == 0:
                    progress_percentage = round((i / total_images) * 100, 2)
                    print()
                    print("-" * 50)
                    print(f"Progress: {progress_percentage}%")
                    print("-" * 50)
                    print()
    else:
        # Original sequential processing
        for index, image in enumerate(dataset.images):
            model_annotations = run_model_on_image(
                image.path,
                model,
                threshold=threshold,
                segment_tolerance=segment_tolerance,
                add_tag=add_tag,
            )
            keep_annotations = annotations_non_max_suppression(
                model_annotations,
                model_names_keys,
                iou_threshold=non_max_suppression_threshold,
                active=non_max_suppression,
            )

            image.annotations = image.annotations + keep_annotations
            if index % 25 == 0:
                progress_percentage = round((index / total_images) * 100, 2)
                print()
                print("-" * 50)
                print(f"Progress: {progress_percentage}%")
                print("-" * 50)
                print()

    dataset.categories = [
        Category(name=name, id=index) for index, name in enumerate(new_categories)
    ]
    return remap_category_ids(dataset)


# TODO: refactor
# pylint: disable=too-many-branches,too-many-statements
def write(
    dataset: Dataset,
    target_dir: str,
    pre_process_images: bool = False,
    remove_existing: bool = False,
    write_original: bool = True,
) -> None:
    raw_path = Path(target_dir) / "raw"
    target_path = Path(target_dir) / "yolo"
    data_yaml_path = target_path / "data.yaml"
    if target_path.exists() and remove_existing:
        if data_yaml_path.exists():
            # remove the directory
            shutil.rmtree(target_path)
            if raw_path.exists():
                shutil.rmtree(raw_path)
        else:
            raise ValueError(
                f"Directory {target_path} already exists, but is not a yolo_v8 dataset"
            )

    target_path.mkdir(parents=True, exist_ok=True)
    data_for_yaml = {
        "names": [category.name for category in dataset.categories],
        "nc": len(dataset.categories),
    }
    for group in dataset.groups:
        data_for_yaml[group] = f"./{group}/images"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data_for_yaml,
            f,
        )
    total_images: dict[str, int] = {}
    for image in dataset.images:
        # Copy the original image
        raw_folder = raw_path / image.group
        if write_original and pre_process_images:
            image_folder = raw_folder / "images"
            image_folder.mkdir(parents=True, exist_ok=True)
            image_path = image_folder / f"{image.id}.jpg"
            shutil.copy(image.path, image_path)

        # write image information
        dataset_path = raw_path / image.group / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        with open(dataset_path / f"{image.id}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(image.__dict__, cls=CustomEncoder, indent=4))

        # write confidences
        confidences_folder = target_path / image.group / "confidences"
        confidences_folder.mkdir(parents=True, exist_ok=True)
        confidence_path = confidences_folder / f"{image.id}.txt"
        with open(confidence_path, "w", encoding="utf-8") as f:
            for annotation in image.annotations:
                f.write(f"{annotation.conf or ''}\n")

        # write image information
        image_info_folder = target_path / image.group / "image_info"
        image_info_folder.mkdir(parents=True, exist_ok=True)
        image_info_path = image_info_folder / f"{image.id}.json"
        with open(image_info_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(image.info, cls=CustomEncoder, indent=4))

        image_folder = target_path / image.group / "images"
        image_folder.mkdir(parents=True, exist_ok=True)
        image_path = image_folder / f"{image.id}.jpg"
        with ImagePil.open(image.path) as img:
            if pre_process_images:
                new_image = preprocess_image(img)
                ImagePil.fromarray(new_image).save(image_path)  # type: ignore
            else:
                img.save(image_path)
        total_images[image.group] = total_images.get(image.group, 0) + 1
        label_folder = target_path / image.group / "labels"
        label_folder.mkdir(parents=True, exist_ok=True)
        label_path = label_folder / f"{image.id}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for annotation in image.annotations:
                if annotation.task == "detect" and annotation.bbox:
                    yolo_bbox = yolov8_from_bbox(annotation.bbox)
                    f.write(f"{annotation.category_id} {' '.join(str(x) for x in yolo_bbox)}\n")
                elif annotation.task in "segment" and annotation.segmentation:
                    polygon_str = " ".join(str(x) for x in annotation.segmentation)
                    f.write(f"{annotation.category_id} {polygon_str}\n")
                elif annotation.task == "obb" and annotation.obb:
                    obb_str = " ".join(str(x) for x in annotation.obb)
                    f.write(f"{annotation.category_id} {obb_str}\n")
                elif annotation.task == "obb" and annotation.segmentation:
                    polygon_str = " ".join(str(x) for x in annotation.segmentation)
                    f.write(f"{annotation.category_id} {polygon_str}\n")
    print()
    print("Dataset saved as yolo_v8 format in: ", target_dir)
    print("total images: ", len(dataset.images))
    print("total images per group:")
    for group, total in total_images.items():
        print(f"\t{group}: {total}")


def check_device() -> str:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


class CustomEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        try:
            return o.__dict__
        except AttributeError:
            pass
        return super().default(o)


def train(
    dataset: Dataset,
    data_file: str,
    model_name: str,
    model_output: str,
    epochs: int = 100,
    batch_size: int = 8,
) -> Dict[str, Any]:
    model = YOLO(model_name)
    device = check_device()
    print("Training on device: ", device)

    results = model.train(
        data=data_file,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        val=True,
    )
    if results is None:
        print("Training failed")
        return {"dataset": dataset}
    try:
        results_dict = {
            key: value for key, value in results.__dict__.items() if not key.startswith("on_")
        }
    # pylint: disable=broad-except
    except Exception:
        results_dict = {
            "box": json.loads(json.dumps(results.box.__dict__ or {}, cls=CustomEncoder)),
            "seg": json.loads(json.dumps(results.seg.__dict__ or {}, cls=CustomEncoder)),
        }

    # Save the model
    current_model_location = Path(results.save_dir) / "weights" / "best.pt"
    model_output_dir = Path(model_output).parent
    model_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(current_model_location, model_output)
    print("Model saved in: ", model_output)
    # results dict and ignore the key starting with on_

    return {
        "dataset": dataset,
        "results": json.loads(json.dumps(results_dict, cls=CustomEncoder)),
        "model_output": model_output,
    }


def find_data_yaml_folder(temp_dir: str) -> str | None:
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith("data.yaml"):
                return root
    return None


def load_from_zip(zip_path: str, temp_dir: str | None = None) -> Dataset:
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    # find data.yaml in the zip inside the temp_dir, recursively
    data_yaml_folder_path = find_data_yaml_folder(temp_dir)
    if not data_yaml_folder_path:
        raise ValueError("data.yaml not found in the zip file")
    return load_dataset(None, data_yaml_folder_path)
