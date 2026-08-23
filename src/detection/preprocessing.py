from __future__ import annotations

import numpy as np


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Return a contiguous BGR frame suitable for YOLO.

    Ultralytics accepts BGR numpy arrays directly; we only ensure layout.
    """
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got shape={getattr(frame, 'shape', None)}")
    return np.ascontiguousarray(frame)
