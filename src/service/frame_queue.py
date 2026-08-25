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
        self._queue: queue.Queue[QueuedFrame] = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._dropped = 0
        self._accepted = 0
        self._closed = threading.Event()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def put(self, item: QueuedFrame) -> None:
        if self._closed.is_set():
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
        self._queue.put(item)
        self._accepted += 1

    def get(self, timeout: float = 1.0) -> QueuedFrame | None:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            return None
        return item

    def close(self) -> None:
        self._closed.set()

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
    released_event: threading.Event | None = None,
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

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.05)
                        continue

                now = time.perf_counter()
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
        finally:
            cap.release()
            if released_event is not None:
                released_event.set()
            frame_queue.close()

    thread = threading.Thread(target=_run, name="capture", daemon=True)
    thread.start()
    return thread
