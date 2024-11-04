from typing import Tuple, Sequence

import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.errors import TopologicalError

from pflows.typedef import Annotation

ROUNDING = 6


def calculate_center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (round((x1 + x2) / 2, ROUNDING), round((y1 + y2) / 2, ROUNDING))


def calculate_center_from_polygon(polygon: Tuple[float, ...]) -> Tuple[float, float] | None:
    if len(polygon) == 0:
        return None
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


def iou_polygons(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    # Convert tuples to numpy arrays
    a_array = np.array(a).reshape(-1, 2)
    b_array = np.array(b).reshape(-1, 2)

    # Create polygon objects
    poly_a = Polygon(a_array)
    poly_b = Polygon(b_array)

    # Ensure polygons are valid
    poly_a = make_valid(poly_a)
    poly_b = make_valid(poly_b)

    try:
        # Calculate intersection and union areas
        intersection_area = float(poly_a.intersection(poly_b).area)
        union_area = float(poly_a.area + poly_b.area - intersection_area)

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou
    except (TopologicalError, ValueError) as e:
        print(f"Error calculating IoU: {e}. This may be due to invalid polygon geometries.")
        return 0.0
    except ZeroDivisionError:
        print("Error calculating IoU: Union area is zero. The polygons might be empty or invalid.")
        return 0.0
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0


def check_polygon_containment(
    polygon1: Tuple[float, ...], polygon2: Tuple[float, ...], threshold=1
):
    """
    Check if polygon1 contains polygon2 and calculate the containment percentage.

    :param polygon1: List of coordinates for polygon1 in format [x1,y1,x2,y2,x3,y3...]
    :param polygon2: List of coordinates for polygon2 in format [x1,y1,x2,y2,x3,y3...]
    :param threshold: Percentage threshold for considering containment (default 100%)
    :return: Tuple (is_contained, containment_percentage)
    """
    # Convert coordinate lists to list of tuples
    polygon1_points = [(polygon1[i], polygon1[i + 1]) for i in range(0, len(polygon1), 2)]
    polygon2_points = [(polygon2[i], polygon2[i + 1]) for i in range(0, len(polygon2), 2)]

    # Create Shapely Polygon objects
    polygon1 = Polygon(polygon1_points)
    polygon2 = Polygon(polygon2_points)

    # Calculate the intersection of the two polygons
    intersection = polygon1.intersection(polygon2)

    # Calculate the containment percentage
    containment_percentage = intersection.area / polygon2.area

    # Check if the containment percentage meets the threshold
    is_contained = containment_percentage >= threshold

    return is_contained, containment_percentage


def check_point_in_polygon(point: Tuple[float, float], polygon: Tuple[float, ...]) -> bool:
    """
    Check if a point lies inside a polygon.

    Args:
        point: Tuple of (x, y) coordinates
        polygon: Tuple of polygon coordinates in format (x1,y1,x2,y2,...)

    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    # Convert polygon coordinates to list of tuples
    polygon_points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

    # Create Shapely Point and Polygon objects
    point_obj = Point(point)
    polygon_obj = Polygon(polygon_points)

    # Check if point is inside or on the boundary of the polygon
    return polygon_obj.contains(point_obj) or polygon_obj.boundary.contains(point_obj)
