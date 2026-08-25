from __future__ import annotations

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.service.backend_client import BackendClient
from src.service.frame_queue import CaptureError
from src.service.runtime import InspectionRuntime
from src.service.video_inspection_job import run_video_inspection_job

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
    except CaptureError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": exc.message},
        )
    return {"status": "running", **body.model_dump()}


@app.post("/camera/stop")
def camera_stop():
    if _runtime is None:
        return {"status": "stopped"}
    _runtime.stop()
    return {"status": "stopped"}


@app.get("/camera/status")
def camera_status():
    if _runtime is None:
        return {"running": False}
    st = _runtime.status()
    return {
        "running": st.running,
        "machine_id": st.machine_id,
        "source": st.source,
        "fps_capture": round(st.fps_capture, 2),
        "fps_inference": round(st.fps_inference, 2),
        "queue_depth": st.queue_depth,
        "dropped_frames": st.dropped_frames,
    }


UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"


@app.post("/inspections/video")
async def inspect_video(
    file: UploadFile = File(...),
    machine_id: str = Form("LOOM-01"),
    video_inspection_id: str = Form(...),
):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{video_inspection_id}_{file.filename}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    backend = BackendClient(base_url=BACKEND_URL, internal_api_key=INTERNAL_API_KEY)
    thread = threading.Thread(
        target=run_video_inspection_job,
        kwargs=dict(
            video_path=dest,
            machine_id=machine_id,
            video_inspection_id=video_inspection_id,
            backend=backend,
        ),
        daemon=True,
    )
    thread.start()
    return {"status": "processing", "video_inspection_id": video_inspection_id}


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
