from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.service.frame_queue import CaptureError
from src.service.runtime import InspectionRuntime, RuntimeConflict

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "change-me")
PORT = int(os.environ.get("PORT", "8100"))
logger.info("backend=%s port=%s", BACKEND_URL, PORT)

_runtime: InspectionRuntime | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _runtime
    _runtime = InspectionRuntime(
        backend_url=BACKEND_URL,
        internal_api_key=INTERNAL_API_KEY,
    )
    yield
    if _runtime is not None:
        _runtime.close()
        _runtime = None


app = FastAPI(title="wastara-model", lifespan=lifespan)


class StartRequest(BaseModel):
    source: str = Field(default="webcam")
    machine_id: str = Field(default="LOOM-01")


@app.post("/camera/start")
def camera_start(body: StartRequest):
    if _runtime is None:
        return {"status": "error", "message": "runtime not ready"}
    try:
        _runtime.start(
            source=body.source,
            machine_id=body.machine_id,
        )
    except RuntimeConflict as exc:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "code": "SESSION_STILL_PROCESSING",
                "state": exc.state,
                "message": "Sesi sebelumnya masih diproses.",
            },
        )
    except CaptureError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": exc.message},
        )
    return {"status": "running", "state": "CAPTURING", **body.model_dump()}


@app.post("/camera/stop")
def camera_stop():
    if _runtime is None:
        return {"status": "stopped", "state": "IDLE"}
    return _status_payload(_runtime.stop())


@app.get("/camera/status")
def camera_status():
    if _runtime is None:
        return {"running": False, "state": "IDLE"}
    return _status_payload(_runtime.status())


def _status_payload(st):
    return {
        "state": st.state,
        "running": st.running,
        "machine_id": st.machine_id,
        "source": st.source,
        "fps_capture": round(st.fps_capture, 2),
        "fps_inference": round(st.fps_inference, 2),
        "queue_depth": st.queue_depth,
        "captured_frames": st.captured_frames,
        "processed_frames": st.processed_frames,
        "remaining_frames": st.remaining_frames,
        "dropped_inference_frames": st.dropped_inference_frames,
        "dropped_frames": st.dropped_inference_frames,
        "failed_frames": st.failed_frames,
        "camera_released": st.camera_released,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.service.app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
