import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.detection.types import BBox
from src.inspection.roi_extractor import GarmentROI
from src.inspection.types import DefectPrediction
from src.service import runtime as runtime_module
from src.service.frame_queue import QueuedFrame
from src.service.runtime import InspectionRuntime, RuntimeConflict
from src.service.streaming_engine import StreamingInspectionEngine


class _AlwaysAnomaly:
    def analyze(self, _roi):
        return DefectPrediction(
            defect_type="anomaly",
            confidence=0.9,
            bbox=BBox(2, 2, 8, 8),
            is_defect=True,
            mode="anomaly",
            score=0.9,
        )


def test_stream_confirms_once_then_finalizes_frame_range(monkeypatch):
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    garment_bbox = BBox(0, 0, 12, 12)
    roi = GarmentROI(image=frame, garment_bbox=garment_bbox)
    tracked = SimpleNamespace(
        track_id=7,
        detection=SimpleNamespace(bbox=garment_bbox),
    )

    engine = StreamingInspectionEngine.__new__(StreamingInspectionEngine)
    engine.detector = object()
    engine.analyzer = _AlwaysAnomaly()
    engine.agg_cfg = SimpleNamespace(confidence_threshold=0.5)
    engine._stream_confidence_threshold = 0.5
    engine._live_min_confirmations = 3
    engine._anomaly_states = {}
    engine._encode_evidence = lambda *_args, **_kwargs: b"jpg"

    monkeypatch.setattr(
        "src.service.streaming_engine.refine_garment_bbox",
        lambda *_args, **_kwargs: garment_bbox,
    )
    monkeypatch.setattr(
        "src.service.streaming_engine.extract_roi",
        lambda *_args, **_kwargs: roi,
    )
    monkeypatch.setattr(
        "src.service.streaming_engine.tighten_roi_to_fabric",
        lambda value: value,
    )

    assert engine._observe_anomaly(tracked, frame, 10, 400.0) is None
    assert engine._observe_anomaly(tracked, frame, 11, 440.0) is None

    confirmed = engine._observe_anomaly(tracked, frame, 12, 480.0)
    assert confirmed is not None
    assert confirmed.track_id == 7
    assert confirmed.frame_start == 10
    assert confirmed.frame_end == 12

    assert engine._observe_anomaly(tracked, frame, 13, 520.0) is None
    finalized = engine._finalize_track(7)
    assert finalized is not None
    assert finalized.frame_start == 10
    assert finalized.frame_end == 13
    assert engine._finalize_track(7) is None


def test_runtime_rejects_restart_while_draining_and_completes(monkeypatch):
    release_capture = threading.Event()
    allow_inference = threading.Event()

    class _Engine:
        def process_frame(self, image, **_kwargs):
            allow_inference.wait(timeout=2.0)
            return image, []

        def flush_pending(self, _frame_id):
            return []

    def _start_capture_thread(*, frame_queue, stop_event, released_event, **_kwargs):
        def _capture():
            for frame_id in range(3):
                frame_queue.put(
                    QueuedFrame(
                        frame_id=frame_id,
                        timestamp_ms=frame_id * 100.0,
                        image=np.zeros((2, 2, 3), dtype=np.uint8),
                    )
                )
            stop_event.wait(timeout=2.0)
            release_capture.wait(timeout=2.0)
            released_event.set()
            frame_queue.close()

        thread = threading.Thread(target=_capture, daemon=True)
        thread.start()
        return thread

    monkeypatch.setattr(runtime_module, "StreamingInspectionEngine", _Engine)
    monkeypatch.setattr(runtime_module, "start_capture_thread", _start_capture_thread)

    runtime = InspectionRuntime(
        backend_url="http://127.0.0.1:1",
        internal_api_key="test",
    )
    monkeypatch.setattr(runtime._backend, "send_frame_jpeg", lambda _frame: None)
    runtime.start(source="fake", machine_id="LOOM-TEST")

    with pytest.raises(RuntimeConflict):
        runtime.start(source="fake", machine_id="LOOM-TEST")

    stopped = runtime.stop()
    assert stopped.state == "DRAINING"
    assert stopped.running is True

    with pytest.raises(RuntimeConflict):
        runtime.start(source="fake", machine_id="LOOM-TEST")

    release_capture.set()
    allow_inference.set()
    assert runtime._worker_done.wait(timeout=2.0)
    completed = runtime.status()
    assert completed.state == "COMPLETED"
    assert completed.running is False
    assert completed.camera_released is True
    assert completed.captured_frames == 3
    assert completed.processed_frames == 3
    assert completed.remaining_frames == 0
    runtime.close()
