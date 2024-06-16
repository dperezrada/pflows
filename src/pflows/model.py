from hashlib import md5
from pathlib import Path
from typing import List

from PIL import Image as ImagePil

from pflows.typedef import Image


def get_image_info(
    image_path: str, group_name: str, intermediate_ids: List[str] | None = None
) -> Image:
    with ImagePil.open(image_path) as img:
        width, height = img.size
        image_bytes = img.tobytes()
        size_bytes = len(image_bytes)
        size_kb = int(round(size_bytes / 1024, 2))
        image_hash = md5(image_bytes).hexdigest()
    intermediate_ids = intermediate_ids or []
    image_path_obj = Path(image_path)
    if image_path_obj.name not in intermediate_ids:
        intermediate_ids.append(image_path_obj.name)
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
