from __future__ import annotations

import queue
import threading
import time
import cv2
from dataclasses import dataclass
from pathlib import Path
from src.tracking.tracker import IoUTracker

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class CaptureError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass
class QueuedFrame:
    frame_id: int
    timestamp_ms: float
    image: np.ndarray


class FrameQueue:
    def __init__(self, maxsize: int = 30) -> None:
        self._queue: queue.Queue[QueuedFrame | None] = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._dropped = 0

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

    def put(self, item: QueuedFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
        self._queue.put(item)

    def get(self, timeout: float = 1.0) -> QueuedFrame | None:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            return None
        return item

    def close(self) -> None:
        self._queue.put(None)

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def resolve_source(raw: str) -> str | int:
    if raw in {"webcam", "0", "default"}:
        return 0
    path = Path(raw)
    if not path.is_absolute():
        candidate = ROOT / raw
        if candidate.exists():
            return str(candidate)
    return raw


def open_capture(source: str):
    resolved = resolve_source(source)
    cap = cv2.VideoCapture(resolved)
    return cap, resolved


def _webcam_hint(resolved: str | int) -> str:
    if resolved != 0:
        return ""
    return (
        " On macOS, grant camera access to the app running wastara-serve "
        "(System Settings → Privacy & Security → Camera), then restart the model service. "
        "Or set NEXT_PUBLIC_INSPECTION_SOURCE to a local video file path."
    )


def is_file_source(source: str) -> bool:
    if source in {"webcam", "0", "default"}:
        return False
    resolved = resolve_source(source)
    if isinstance(resolved, int):
        return False
    return Path(resolved).is_file()


def start_event_capture_thread(
    *,
    source: str,
    frame_queue: FrameQueue,
    stop_event: threading.Event,
    detector,
    tracking_cfg: dict | None = None,
    on_frame=None,
) -> threading.Thread:
    cap, resolved = open_capture(source)
    if not cap.isOpened():
        raise CaptureError(
            f"Failed to open video source {source!r}.{_webcam_hint(resolved)}"
        )

    is_video = is_file_source(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cfg = tracking_cfg or {}
    gate_tracker = IoUTracker(
        match_iou=float(cfg.get("match_iou", 0.3)),
        track_buffer=int(cfg.get("track_buffer", 30)),
        min_hits=int(cfg.get("min_hits", 2)),
        track_threshold=float(cfg.get("track_threshold", 0.4)),
    )

    def _run() -> None:
        frame_id = 0

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

            timestamp_ms = (
                (frame_id / video_fps) * 1000.0 if is_video else time.perf_counter() * 1000.0
            )

            if on_frame is not None:
                on_frame(frame)

            detections = detector.predict(frame)
            tracks = gate_tracker.update(
                detections, frame_id=frame_id, timestamp_ms=timestamp_ms
            )

            if tracks:
                frame_queue.put(
                    QueuedFrame(
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        image=frame.copy(),
                    )
                )

            frame_id += 1

        cap.release()
        frame_queue.close()

    thread = threading.Thread(target=_run, name="event-capture", daemon=True)
    thread.start()
    return thread

