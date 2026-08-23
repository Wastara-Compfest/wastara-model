from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    machine_id: str
    anomaly_score: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    frame_start: int
    frame_end: int


class BackendClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_api_key: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_api_key = internal_api_key
        self._http = httpx.Client(timeout=15.0)
        self._ws = None

    def close(self) -> None:
        self._http.close()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def post_anomaly(self, event: AnomalyEvent, evidence_jpg: bytes) -> str | None:
        payload: dict[str, Any] = {
            "machine_id": event.machine_id,
            "anomaly_score": round(event.anomaly_score, 4),
            "bbox": {
                "x": event.bbox_x,
                "y": event.bbox_y,
                "w": event.bbox_w,
                "h": event.bbox_h,
            },
            "frames": {"start": event.frame_start, "end": event.frame_end},
        }
        try:
            res = self._http.post(
                f"{self.base_url}/internal/defect-events",
                headers={"X-Internal-Key": self.internal_api_key},
                data={"data": json.dumps(payload)},
                files={"evidence": ("evidence.jpg", evidence_jpg, "image/jpeg")},
            )
            res.raise_for_status()
            body = res.json()
            return str(body.get("id")) if isinstance(body, dict) else None
        except Exception as exc:
            logger.warning("failed to post defect event: %s", exc)
            return None

    def send_frame_jpeg(self, frame: np.ndarray, quality: int = 72) -> None:
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return
        jpeg = buf.tobytes()
        try:
            ws_url = self.base_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            if self._ws is None:
                import websocket

                self._ws = websocket.create_connection(
                    f"{ws_url}/internal/ws/frames",
                    header=[f"X-Internal-Key: {self.internal_api_key}"],
                )
            self._ws.send(jpeg, opcode=0x2)
        except Exception as exc:
            logger.warning("ws frame send failed: %s", exc)
            self._ws = None
