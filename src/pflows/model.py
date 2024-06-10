from hashlib import md5
from typing import List, Sequence

from PIL import Image as ImagePil

from pflows.typedef import Image


def get_image_info(image_path: str, group_name: str, intermediate_ids: List[str]|None = None) -> Image:
    with ImagePil.open(image_path) as img:
        width, height = img.size
        image_bytes = img.tobytes()
        size_bytes = len(image_bytes)
        size_kb = int(round(size_bytes / 1024, 2))
        image_hash = md5(image_bytes).hexdigest()
    image: Image = Image(
        id=image_hash,
        intermediate_ids=intermediate_ids or [],
        path=str(image_path),
        width=width,
        height=height,
        size_kb=size_kb,
        group=group_name,
    )
    return image
