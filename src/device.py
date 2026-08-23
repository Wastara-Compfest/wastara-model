from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Return a torch/ultralytics device string."""
    key = (requested or "auto").strip().lower()

    if key == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if key in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"

    if key == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return "mps"

    if key == "cpu":
        return "cpu"

    # allow cuda:0 style
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device '{requested}' requested but CUDA not available")
        return requested

    raise ValueError(f"unsupported device: {requested}")
