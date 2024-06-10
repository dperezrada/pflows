from dataclasses import asdict
import tempfile
from typing import List
from PIL import Image as ImagePil
import math

from pflows.model import get_image_info
from pflows.polygons import calculate_center_from_bbox
from pflows.typedef import Annotation, Dataset, Image


def crop_to_annotations(dataset: Dataset) -> Dataset:
    """
    Crop images to match the min and max x/y of the annotations.
    """
    temp_images_path = tempfile.mkdtemp()
    new_images: List[Image] = []
    for image in dataset.images:
        pilImage = ImagePil.open(image.path)
        width, height = pilImage.size
        min_x = 1000000
        min_y = 1000000
        max_x = 0
        max_y = 0

        for annotation in image.annotations:
            if annotation.bbox is None:
                continue
            min_x = min(int(min(annotation.bbox[0::2])*width), min_x)
            min_y = min(int(min(annotation.bbox[1::2])*height), min_y)
            max_x = max(int(max(annotation.bbox[0::2])*width), max_x)
            max_y = max(int(max(annotation.bbox[1::2])*height), max_y)
        new_width = int((max_x - min_x))
        new_height = int((max_y - min_y))
        # crop the image
        # Adjust annotations for the cropped image
        for annotation in image.annotations:
            if annotation.bbox is not None:
                new_bbox = tuple((width*coord - min_x)/new_width if i % 2 == 0 else (height*coord - min_y)/new_height for i, coord in enumerate(annotation.bbox))
                annotation.bbox = new_bbox # type: ignore
            
            if annotation.center is not None:
                new_center = tuple((width*coord - min_x)/new_width if i % 2 == 0 else (height*coord - min_y)/new_height for i, coord in enumerate(annotation.center))
                annotation.center = new_center
            
            if annotation.segmentation is not None:
                new_segmentation = tuple((width*coord - min_x)/new_width if i % 2 == 0 else (height*coord - min_y)/new_height for i, coord in enumerate(annotation.segmentation))
                annotation.segmentation = new_segmentation
        
        croppedPilImage = pilImage.crop((
            min_x,
            min_y,
            max_x,
            max_y,
        ))
        new_path = f"{temp_images_path}/{image.id}.jpg"
        croppedPilImage.save(new_path)
        new_image = get_image_info(new_path, image.group, image.intermediate_ids + [image.id])
        new_images.append(
            Image(
                **{
                    **asdict(image),
                    **asdict(new_image),
                    "annotations": image.annotations
                }
            )
        )
    return Dataset(
        images=new_images,
        categories=dataset.categories,
        groups=dataset.groups
    )

    

def get_annotations_from_sliced_image(sliced_image: Image, original_image: Image, crop_x: int, crop_y: int) -> List[Annotation]:
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
            if bbox_x2 <= crop_x or bbox_x1 >= crop_x + w or bbox_y2 <= crop_y or bbox_y1 >= crop_y + h:
                continue

            # Calculate the intersection of the bounding box with the sliced image
            intersection_x1 = max(bbox_x1, crop_x)
            intersection_y1 = max(bbox_y1, crop_y)
            intersection_x2 = min(bbox_x2, crop_x + w)
            intersection_y2 = min(bbox_y2, crop_y + h)

            # Calculate the adjusted bounding box coordinates
            adjusted_bbox_x1 = max(0, intersection_x1 - crop_x) / w
            adjusted_bbox_y1 = max(0, intersection_y1 - crop_y) / h
            adjusted_bbox_x2 = min(w, intersection_x2 - crop_x) / w
            adjusted_bbox_y2 = min(h, intersection_y2 - crop_y) / h

            sliced_annotation = Annotation(
                id=annotation.id,
                category_id=annotation.category_id,
                center=calculate_center_from_bbox((adjusted_bbox_x1, adjusted_bbox_y1, adjusted_bbox_x2, adjusted_bbox_y2)),
                bbox=(adjusted_bbox_x1, adjusted_bbox_y1, adjusted_bbox_x2, adjusted_bbox_y2),
                segmentation=annotation.segmentation,
                task=annotation.task,
                conf=annotation.conf,
                category_name=annotation.category_name,
                tags=annotation.tags
            )
            sliced_annotations.append(sliced_annotation)

    return sliced_annotations

def slice_image(image: Image, slice_height: int = 640, slice_width: int = 640,
                min_overlap_height_ratio: float = 0.1, min_overlap_width_ratio: float = 0.1) -> List[Image]:

    temp_folder = tempfile.mkdtemp()
    # Only detect is supported for now
    if not all([
        annotation.task == "detect"
        for annotation in image.annotations
    ]):
        return [image]

    sliced_images = []
    img = ImagePil.open(image.path)

    # Calculate the number of slices in each dimension
    num_slices_height = math.ceil(image.height / (slice_height * (1 - min_overlap_height_ratio)))
    num_slices_width = math.ceil(image.width / (slice_width * (1 - min_overlap_width_ratio)))

    if num_slices_height == 1 and num_slices_width == 1:
        return [image]
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
            sliced_path = f"{temp_folder}/{image.id}_{left}_{top}.jpg"
            sliced_img.save(sliced_path)
            sliced_image = Image(
                id=f"{image.id}_{len(sliced_images)}",
                path=sliced_path,
                intermediate_ids=image.intermediate_ids,
                width=slice_width,
                height=slice_height,
                size_kb=sliced_img.size[0] * sliced_img.size[1] // 1024,
                group=image.group,
                tags=image.tags
            )
            annotations = get_annotations_from_sliced_image(sliced_image, image, left, top)
            sliced_image.annotations = annotations
            sliced_images.append(sliced_image)
    return sliced_images

def slice_dataset(dataset: Dataset, slice_height: int = 1024, slice_width: int = 1024, overlap_height_ratio: float = 0.1, overlap_width_ratio: float = 0.1) -> Dataset:
    sliced_images = []
    for image in dataset.images:
        sliced_images.extend(slice_image(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio))
    return Dataset(
        images=sliced_images,
        categories=dataset.categories,
        groups=dataset.groups
    )