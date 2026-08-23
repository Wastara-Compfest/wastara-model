from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.reader import VideoReader


def _write_dummy_video(path: Path, frames: int = 5, fps: int = 10, size=(64, 48)) -> None:
    w, h = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 20) % 255
        writer.write(frame)
    writer.release()


def test_video_reader_valid(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    _write_dummy_video(video)
    with VideoReader(video) as reader:
        meta = reader.meta
        assert meta.width == 64
        assert meta.height == 48
        assert meta.fps > 0
        collected = list(reader.frames(frame_skip=1))
        assert len(collected) >= 1
        frame_id, ts, frame = collected[0]
        assert frame_id == 0
        assert ts == 0
        assert frame.shape == (48, 64, 3)


def test_video_reader_missing():
    with pytest.raises(FileNotFoundError):
        VideoReader("/tmp/does_not_exist_wastara.mp4")


def test_video_reader_frame_skip(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    _write_dummy_video(video, frames=6)
    with VideoReader(video) as reader:
        ids = [fid for fid, _, _ in reader.frames(frame_skip=2)]
    assert ids == [i for i in ids if i % 2 == 0]
