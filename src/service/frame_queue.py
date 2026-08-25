from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
    """Open capture on the caller thread (required for macOS camera permission)."""
    import cv2

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


def start_capture_thread(
    *,
    source: str,
    frame_queue: FrameQueue,
    stop_event: threading.Event,
    target_fps: float = 24.0,
) -> threading.Thread:
    import cv2

    cap, resolved = open_capture(source)
    if not cap.isOpened():
        raise CaptureError(
            f"Failed to open video source {source!r}.{_webcam_hint(resolved)}"
        )

    def _run() -> None:
        frame_id = 0
        interval = 1.0 / max(1.0, target_fps)
        last = time.perf_counter()

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

            now = time.perf_counter()
            last = now
            frame_queue.put(
                QueuedFrame(
                    frame_id=frame_id,
                    timestamp_ms=frame_id * interval * 1000.0,
                    image=frame.copy(),
                )
            )
            frame_id += 1

            sleep_for = interval - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        frame_queue.close()

    thread = threading.Thread(target=_run, name="capture", daemon=True)
    thread.start()
    return thread


def is_file_source(source: str) -> bool:
    """Check if the source is a local file path (not webcam)."""
    if source in {"webcam", "0", "default"}:
        return False
    resolved = resolve_source(source)
    if isinstance(resolved, int):
        return False
    return Path(resolved).is_file()


def start_smart_capture_thread(
    *,
    source: str,
    frame_queue: FrameQueue,
    stop_event: threading.Event,
    stream_fps: float = 6.0,
    idle_check_interval: float = 0.15,
) -> threading.Thread:
    """Smart capture: use YOLO to detect objects first, only queue frames with
    detected garments. Idle frames (no object) are skipped, saving downstream
    processing. A lightweight visualization is still sent for empty frames so the
    user sees progress. For video files this processes much faster than real-time."""
    import cv2
    from ultralytics import YOLO

    cap, resolved = open_capture(source)
    if not cap.isOpened():
        raise CaptureError(
            f"Failed to open video source {source!r}.{_webcam_hint(resolved)}"
        )

    # Load a lightweight YOLO model for gate detection
    gate_model_path = ROOT / "models" / "clothing" / "finetuned" / "yolov8n-conveyor-clothing.pt"
    if not gate_model_path.exists():
        gate_model_path = ROOT / "yolov8n.pt"

    gate_model = YOLO(str(gate_model_path))

    is_video = is_file_source(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    def _run() -> None:
        frame_id = 0
        queued_count = 0
        interval = 1.0 / max(1.0, stream_fps)

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    # End of video file — signal completion
                    break
                # Webcam glitch — try to recover
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

            timestamp_ms = (frame_id / video_fps) * 1000.0

            # Run lightweight YOLO detection as gate
            results = gate_model.predict(
                source=frame,
                conf=0.3,
                imgsz=320,  # Small size for speed
                verbose=False,
            )
            has_object = False
            if results and len(results) > 0:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    has_object = True

            if has_object:
                frame_queue.put(
                    QueuedFrame(
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        image=frame.copy(),
                    )
                )
                queued_count += 1

            frame_id += 1

            # For video files, process as fast as possible but still yield
            if is_video:
                if frame_id % 10 == 0:
                    time.sleep(0.001)  # yield to other threads
            else:
                sleep_for = interval - idle_check_interval
                if sleep_for > 0:
                    time.sleep(sleep_for)

        cap.release()
        frame_queue.close()

    thread = threading.Thread(target=_run, name="smart-capture", daemon=True)
    thread.start()
    return thread

