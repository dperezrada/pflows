from typing import Tuple, Sequence

from shapely.geometry import Polygon, MultiPolygon

from pflows.typedef import Annotation

ROUNDING = 6


def calculate_center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (round((x1 + x2) / 2, ROUNDING), round((y1 + y2) / 2, ROUNDING))


def calculate_center_from_polygon(polygon: Tuple[float, ...]) -> Tuple[float, float]:
    x = [polygon[i] for i in range(0, len(polygon), 2)]
    y = [polygon[i] for i in range(1, len(polygon), 2)]
    return (round(sum(x) / len(x), ROUNDING), round(sum(y) / len(y), ROUNDING))


def polygon_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, ...]:
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2, y1, x2, y2, x1, y2)


def bbox_from_polygon(polygon: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    x = [polygon[i] for i in range(0, len(polygon), 2)]
    y = [polygon[i] for i in range(1, len(polygon), 2)]
    return (min(x), min(y), max(x), max(y))


def get_bbox(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    return (x1, y1, x2, y2)


def get_min_max_bbox(annotations: list[Annotation]) -> tuple[float, float, float, float] | None:
    if all(annotation.bbox is None for annotation in annotations):
        return None
    x1 = min(annotation.bbox[0] for annotation in annotations if annotation.bbox is not None)
    y1 = min(annotation.bbox[1] for annotation in annotations if annotation.bbox is not None)
    x2 = max(annotation.bbox[2] for annotation in annotations if annotation.bbox is not None)
    y2 = max(annotation.bbox[3] for annotation in annotations if annotation.bbox is not None)
    return get_bbox(x1, y1, x2, y2)


def get_biggest_bbox(
    bbox: Tuple[float, float, float, float], new_bbox: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1_new, y1_new, x2_new, y2_new = new_bbox
    return get_bbox(min(x1, x1_new), min(y1, y1_new), max(x2, x2_new), max(y2, y2_new))


def merge_polygons(polygons: Sequence[Tuple[float, ...]]) -> Tuple[float, ...]:
    all_polygons = [Polygon(zip(polygon[::2], polygon[1::2])) for polygon in polygons]
    merged_polygon = all_polygons[0]
    for polygon in all_polygons[1:]:
        merged_polygon = merged_polygon.union(polygon)
    # Check if the merged polygon is a MultiPolygon
    if isinstance(merged_polygon, MultiPolygon):
        # If it's a MultiPolygon, merge all the polygons into a single polygon
        merged_polygon = Polygon(merged_polygon.convex_hull)

    # Convert the merged polygon back to the format (x1, y1, x2, y2, ...)
    merged_coords = list(merged_polygon.exterior.coords)
    merged_coords = [coord for point in merged_polygon.exterior.coords for coord in point]
    return tuple(float(coord) for coord in merged_coords)
