from __future__ import annotations

from src.detection.types import BBox, Detection


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def validate_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> BBox | None:
    """Clamp bbox to image bounds and ensure x1 < x2, y1 < y2."""
    x1 = clamp(float(x1), 0, width - 1)
    y1 = clamp(float(y1), 0, height - 1)
    x2 = clamp(float(x2), 0, width - 1)
    y2 = clamp(float(y2), 0, height - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def postprocess_detections(
    raw: list[dict],
    *,
    class_mapping: dict[int, str],
    garment_class_ids: list[int],
    confidence_threshold: float,
    image_width: int,
    image_height: int,
    keep_garment_only: bool = False,
) -> list[Detection]:
    """Filter by confidence, validate bbox, map class names."""
    out: list[Detection] = []

    for item in raw:
        conf = float(item.get("confidence", 0.0))
        if conf < confidence_threshold:
            continue

        class_id = int(item["class_id"])
        is_garment = class_id in garment_class_ids if garment_class_ids else True
        if keep_garment_only and garment_class_ids and not is_garment:
            continue

        bbox = validate_bbox(
            item["x1"],
            item["y1"],
            item["x2"],
            item["y2"],
            image_width,
            image_height,
        )
        if bbox is None:
            continue

        class_name = class_mapping.get(class_id) or str(
            item.get("class_name") or f"class_{class_id}"
        )

        out.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                is_garment=is_garment,
            )
        )

    out.sort(key=lambda d: d.confidence, reverse=True)
    return out
