from __future__ import annotations

from dataclasses import dataclass, field

from src.detection.types import BBox, Detection
from src.inspection.types import TrackedDetection


def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    class_name: str
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    active: bool = True


class IoUTracker:
    """Lightweight multi-object tracker (IoU association).

    Used as a reliable standalone tracker. For ByteTrack, see ByteTrackTracker.
    """

    def __init__(
        self,
        *,
        match_iou: float = 0.3,
        track_buffer: int = 30,
        min_hits: int = 2,
        track_threshold: float = 0.4,
    ) -> None:
        self.match_iou = match_iou
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.track_threshold = track_threshold
        self._next_id = 1
        self._tracks: list[_Track] = []

    def update(
        self,
        detections: list[Detection],
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> list[TrackedDetection]:
        dets = [d for d in detections if d.confidence >= self.track_threshold]
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            if not track.active and track.time_since_update > self.track_buffer:
                continue
            for di, det in enumerate(dets):
                score = _iou(track.bbox, det.bbox)
                if score >= self.match_iou:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: list[tuple[int, int]] = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matched.append((ti, di))

        out: list[TrackedDetection] = []
        for ti, di in matched:
            track = self._tracks[ti]
            det = dets[di]
            track.bbox = det.bbox
            track.class_name = det.class_name
            track.hits += 1
            track.time_since_update = 0
            track.active = True
            if track.hits >= self.min_hits:
                out.append(
                    TrackedDetection(
                        track_id=track.track_id,
                        detection=det,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                    )
                )

        for di, det in enumerate(dets):
            if di in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks.append(
                _Track(
                    track_id=tid,
                    bbox=det.bbox,
                    class_name=det.class_name,
                    hits=1,
                    age=1,
                    time_since_update=0,
                )
            )
            if self.min_hits <= 1:
                out.append(
                    TrackedDetection(
                        track_id=tid,
                        detection=det,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                    )
                )

        # prune stale
        self._tracks = [
            t
            for t in self._tracks
            if t.time_since_update <= self.track_buffer
        ]
        return out


class ByteTrackTracker:
    """ByteTrack via Ultralytics tracker on an already-loaded YOLO model.

    Garment class_id from YOLO is NOT the track_id.
    """

    def __init__(
        self,
        yolo_model,
        *,
        track_threshold: float = 0.4,
        conf: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
        class_mapping: dict[int, str] | None = None,
    ) -> None:
        self.model = yolo_model
        self.track_threshold = track_threshold
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.class_mapping = class_mapping or {}

    def update(
        self,
        frame,
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> list[TrackedDetection]:
        import numpy as np

        from src.detection.types import BBox, Detection

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        ids = boxes.id
        if ids is None:
            # tracker not ready yet
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        track_ids = ids.cpu().numpy().astype(int)

        out: list[TrackedDetection] = []
        for i in range(len(xyxy)):
            conf = float(confs[i])
            if conf < self.track_threshold:
                continue
            class_id = int(clss[i])
            x1, y1, x2, y2 = xyxy[i].tolist()
            det = Detection(
                class_id=class_id,
                class_name=self.class_mapping.get(class_id, f"class_{class_id}"),
                confidence=conf,
                bbox=BBox(x1, y1, x2, y2),
                is_garment=True,
            )
            out.append(
                TrackedDetection(
                    track_id=int(track_ids[i]),
                    detection=det,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                )
            )
        return out


def create_tracker(
    name: str,
    *,
    yolo_model=None,
    match_iou: float = 0.3,
    track_buffer: int = 30,
    min_hits: int = 2,
    track_threshold: float = 0.4,
    conf: float = 0.5,
    iou: float = 0.45,
    device: str = "cpu",
    imgsz: int = 640,
    class_mapping: dict[int, str] | None = None,
):
    key = (name or "iou").lower()
    if key == "bytetrack":
        if yolo_model is None:
            raise ValueError("bytetrack requires a loaded YOLO model")
        return ByteTrackTracker(
            yolo_model,
            track_threshold=track_threshold,
            conf=conf,
            iou=iou,
            device=device,
            imgsz=imgsz,
            class_mapping=class_mapping,
        )
    return IoUTracker(
        match_iou=match_iou,
        track_buffer=track_buffer,
        min_hits=min_hits,
        track_threshold=track_threshold,
    )
