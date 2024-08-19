import os
import json
from hashlib import md5
from pathlib import Path
from typing import List

import imagesize

from pflows.typedef import Annotation, Image


def get_image_info_fast(image_path: str):
    # Get file size without reading the entire file
    size_bytes = os.path.getsize(image_path)
    size_kb = int(round(size_bytes / 1024, 2))

    # Open the image and get dimensions without loading the entire image into memory
    width, height = imagesize.get(image_path)

    # Compute MD5 hash of the file without loading it entirely into memory
    hasher = md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    image_hash = hasher.hexdigest()

    return width, height, size_kb, image_hash


def get_image_info(
    image_path: str, group_name: str, intermediate_ids: List[str] | None = None
) -> Image:
    width, height, size_kb, image_hash = get_image_info_fast(image_path)

    intermediate_ids = intermediate_ids or []
    image_path_obj = Path(image_path)
    if image_path_obj.name not in intermediate_ids:
        intermediate_ids.append(image_path_obj.name)
    if image_path_obj.stem not in intermediate_ids and image_path_obj.suffix in [
        "jpg",
        "jpeg",
        "png",
    ]:
        intermediate_ids.append(image_path_obj.stem)
    # check if raw folder exists
    image_stem = image_path_obj.stem
    raw_info_path = Path(
        str(image_path_obj)
        .replace("/yolo", "/raw")
        .replace(f"/images/{image_stem}", f"/dataset/{image_stem}")
    )
    raw_info_path = raw_info_path.with_suffix(".json")
    if raw_info_path.exists():
        return Image.from_dict(json.loads(raw_info_path.read_text()))

    image: Image = Image(
        id=image_hash,
        intermediate_ids=intermediate_ids,
        path=str(image_path),
        width=width,
        height=height,
        size_kb=size_kb,
        group=group_name,
    )
    return image


def image_from_image_path(
    path: str,
    annotations: List[Annotation] | None = None,
) -> Image:
    image = get_image_info(path, "train")
    image.annotations = annotations or []

    return image
