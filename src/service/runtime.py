from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from src.service.backend_client import AnomalyEvent, BackendClient
from src.service.frame_queue import CaptureError, FrameQueue, start_event_capture_thread
from src.service.streaming_engine import StreamingInspectionEngine

logger = logging.getLogger(__name__)


@dataclass
class RuntimeStatus:
    running: bool
    machine_id: str | None
    source: str | None
    fps_capture: float
    fps_inference: float
    queue_depth: int
    dropped_frames: int


class InspectionRuntime:
    def __init__(
        self,
        *,
        backend_url: str,
        internal_api_key: str,
        queue_size: int = 30,
    ) -> None:
        self._backend = BackendClient(
            base_url=backend_url,
            internal_api_key=internal_api_key,
        )
        self._queue = FrameQueue(maxsize=queue_size)
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._engine: StreamingInspectionEngine | None = None
        self._machine_id: str | None = None
        self._source: str | None = None
        self._infer_count = 0
        self._infer_t0 = time.perf_counter()
        self._lock = threading.Lock()

    def start(self, *, source: str, machine_id: str) -> None:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                self.stop()

            self._stop.clear()
            self._machine_id = machine_id
            self._source = source
            self._infer_count = 0
            self._infer_t0 = time.perf_counter()
            self._queue = FrameQueue(maxsize=30)
            self._engine = StreamingInspectionEngine()

            try:
                self._capture_thread = start_event_capture_thread(
                    source=source,
                    frame_queue=self._queue,
                    stop_event=self._stop,
                    detector=self._engine.detector,
                    tracking_cfg=self._engine.cfg.get("tracking", {}),
                )
            except CaptureError:
                self._engine = None
                self._machine_id = None
                self._source = None
                raise

            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="inference",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self._queue.close()
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
            if self._capture_thread:
                self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
            self._worker_thread = None
            self._engine = None
            self._machine_id = None
            self._source = None
            self._stop.clear()

    def status(self) -> RuntimeStatus:
        now = time.perf_counter()
        inf_elapsed = max(now - self._infer_t0, 1e-6)
        running = bool(
            self._worker_thread and self._worker_thread.is_alive() and not self._stop.is_set()
        )
        return RuntimeStatus(
            running=running,
            machine_id=self._machine_id,
            source=self._source,
            fps_capture=0.0,
            fps_inference=self._infer_count / inf_elapsed,
            queue_depth=self._queue.depth,
            dropped_frames=self._queue.dropped,
        )

    def _worker_loop(self) -> None:
        engine = self._engine
        if engine is None:
            return

        while not self._stop.is_set():
            item = self._queue.get(timeout=0.5)
            if item is None:
                if self._stop.is_set():
                    break
                continue

            self._process_item(engine, item)

        while True:
            item = self._queue.get(timeout=0.01)
            if item is None:
                break
            self._process_item(engine, item)

        machine_id = self._machine_id or "LOOM-01"
        for alert in engine.flush_pending(max(self._infer_count, 0)):
            event = AnomalyEvent(
                machine_id=machine_id,
                anomaly_score=alert.anomaly_score,
                bbox_x=alert.bbox_x,
                bbox_y=alert.bbox_y,
                bbox_w=alert.bbox_w,
                bbox_h=alert.bbox_h,
                frame_start=alert.frame_id,
                frame_end=alert.frame_id,
            )
            self._backend.post_anomaly(event, alert.evidence_jpg)

    def _process_item(self, engine: StreamingInspectionEngine, item) -> None:
        try:
            vis, alerts = engine.process_frame(
                item.image,
                frame_id=item.frame_id,
                timestamp_ms=item.timestamp_ms,
            )
        except Exception:
            logger.exception("frame processing failed")
            return

        self._infer_count += 1
        self._backend.send_frame_jpeg(vis)

        machine_id = self._machine_id or "LOOM-01"
        for alert in alerts:
            event = AnomalyEvent(
                machine_id=machine_id,
                anomaly_score=alert.anomaly_score,
                bbox_x=alert.bbox_x,
                bbox_y=alert.bbox_y,
                bbox_w=alert.bbox_w,
                bbox_h=alert.bbox_h,
                frame_start=alert.frame_id,
                frame_end=alert.frame_id,
            )
            self._backend.post_anomaly(event, alert.evidence_jpg)

    def close(self) -> None:
        self.stop()
        self._backend.close()
