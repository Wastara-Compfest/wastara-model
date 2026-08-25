from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass

from src.service.backend_client import AnomalyEvent, BackendClient
from src.service.frame_queue import FrameQueue, start_capture_thread
from src.service.streaming_engine import StreamAlert, StreamingInspectionEngine

logger = logging.getLogger(__name__)

ACTIVE_STATES = {"STARTING", "CAPTURING", "DRAINING"}


class RuntimeConflict(Exception):
    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__(f"inspection session is still {state.lower()}")


@dataclass
class RuntimeStatus:
    state: str
    running: bool
    machine_id: str | None
    source: str | None
    fps_capture: float
    fps_inference: float
    queue_depth: int
    captured_frames: int
    processed_frames: int
    remaining_frames: int
    dropped_inference_frames: int
    failed_frames: int
    camera_released: bool


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
        self._queue_size = queue_size
        self._queue = FrameQueue(maxsize=queue_size)
        self._capture_stop = threading.Event()
        self._worker_done = threading.Event()
        self._worker_done.set()
        self._camera_released = threading.Event()
        self._camera_released.set()
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._engine: StreamingInspectionEngine | None = None
        self._machine_id: str | None = None
        self._source: str | None = None
        self._session_token: str | None = None
        self._processed_frames = 0
        self._failed_frames = 0
        self._infer_t0 = time.perf_counter()
        self._state = "IDLE"
        self._lock = threading.Lock()

    def start(self, *, source: str, machine_id: str) -> None:
        with self._lock:
            if self._state in ACTIVE_STATES:
                raise RuntimeConflict(self._state)

            self._state = "STARTING"
            self._capture_stop = threading.Event()
            self._worker_done = threading.Event()
            self._camera_released = threading.Event()
            self._queue = FrameQueue(maxsize=self._queue_size)
            self._engine = None
            self._machine_id = machine_id
            self._source = source
            self._session_token = str(uuid.uuid4())
            self._processed_frames = 0
            self._failed_frames = 0
            self._infer_t0 = time.perf_counter()

            try:
                self._engine = StreamingInspectionEngine()
                capture_thread = start_capture_thread(
                    source=source,
                    frame_queue=self._queue,
                    stop_event=self._capture_stop,
                    released_event=self._camera_released,
                    target_fps=10.0,
                )
            except Exception:
                self._state = "FAILED"
                self._engine = None
                self._machine_id = None
                self._source = None
                self._session_token = None
                self._worker_done.set()
                self._camera_released.set()
                raise

            engine = self._engine
            session_token = self._session_token
            assert engine is not None
            assert session_token is not None

            worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(
                    engine,
                    self._queue,
                    machine_id,
                    session_token,
                    self._worker_done,
                ),
                name="inference",
                daemon=True,
            )
            self._capture_thread = capture_thread
            self._worker_thread = worker_thread
            worker_thread.start()
            self._state = "CAPTURING"

    def stop(self) -> RuntimeStatus:
        with self._lock:
            if self._state in {"CAPTURING", "STARTING"}:
                self._state = "DRAINING"
                self._capture_stop.set()
            elif self._state not in {"DRAINING", "COMPLETED", "FAILED"}:
                self._state = "IDLE"
        return self.status()

    def status(self) -> RuntimeStatus:
        with self._lock:
            state = self._state
            machine_id = self._machine_id
            source = self._source
            processed_frames = self._processed_frames
            failed_frames = self._failed_frames
            queue = self._queue
            camera_released = self._camera_released.is_set()
            infer_t0 = self._infer_t0

        elapsed = max(time.perf_counter() - infer_t0, 1e-6)
        remaining = queue.depth
        captured = queue.accepted
        return RuntimeStatus(
            state=state,
            running=state in {"CAPTURING", "DRAINING"},
            machine_id=machine_id,
            source=source,
            fps_capture=0.0,
            fps_inference=processed_frames / elapsed,
            queue_depth=remaining,
            captured_frames=captured,
            processed_frames=processed_frames,
            remaining_frames=remaining,
            dropped_inference_frames=queue.dropped,
            failed_frames=failed_frames,
            camera_released=camera_released,
        )

    def _worker_loop(
        self,
        engine: StreamingInspectionEngine,
        frame_queue: FrameQueue,
        machine_id: str,
        session_token: str,
        worker_done: threading.Event,
    ) -> None:
        try:
            while True:
                item = frame_queue.get(timeout=0.5)
                if item is None:
                    if frame_queue.closed:
                        break
                    continue
                self._process_item(engine, item, machine_id, session_token)

            for alert in engine.flush_pending(self._processed_frames):
                self._publish_alert(alert, machine_id, session_token)
        except Exception:
            logger.exception("inference worker failed")
            with self._lock:
                if self._session_token == session_token:
                    self._state = "FAILED"
                    self._capture_stop.set()
        else:
            with self._lock:
                if self._session_token == session_token:
                    self._state = "COMPLETED"
        finally:
            worker_done.set()

    def _process_item(
        self,
        engine: StreamingInspectionEngine,
        item,
        machine_id: str,
        session_token: str,
    ) -> None:
        try:
            vis, alerts = engine.process_frame(
                item.image,
                frame_id=item.frame_id,
                timestamp_ms=item.timestamp_ms,
            )
        except Exception:
            logger.exception("frame processing failed")
            with self._lock:
                self._failed_frames += 1
            return

        with self._lock:
            self._processed_frames += 1
        self._backend.send_frame_jpeg(vis)
        for alert in alerts:
            self._publish_alert(alert, machine_id, session_token)

    def _publish_alert(
        self,
        alert: StreamAlert,
        machine_id: str,
        session_token: str,
    ) -> None:
        event = AnomalyEvent(
            event_key=f"{session_token}:{alert.track_id}",
            machine_id=machine_id,
            anomaly_score=alert.anomaly_score,
            bbox_x=alert.bbox_x,
            bbox_y=alert.bbox_y,
            bbox_w=alert.bbox_w,
            bbox_h=alert.bbox_h,
            frame_start=alert.frame_start,
            frame_end=alert.frame_end,
        )
        self._backend.post_anomaly(event, alert.evidence_jpg)

    def close(self) -> None:
        self.stop()
        if not self._worker_done.wait(timeout=30.0):
            logger.warning("inference worker did not finish before shutdown timeout")
        self._backend.close()
