from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchStats:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ious: list[float] | None = None

    def __post_init__(self) -> None:
        if self.ious is None:
            self.ious = []

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        assert self.ious is not None
        return sum(self.ious) / len(self.ious) if self.ious else 0.0


def match_detections(
    preds: list[Box],
    gts: list[Box],
    *,
    iou_threshold: float = 0.5,
) -> MatchStats:
    """Greedy one-to-one matching by highest IoU."""
    stats = MatchStats()
    if not gts and not preds:
        return stats
    if not gts:
        stats.false_positives = len(preds)
        return stats
    if not preds:
        stats.false_negatives = len(gts)
        return stats

    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            score = iou(p, g)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    used_p: set[int] = set()
    used_g: set[int] = set()
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        stats.true_positives += 1
        assert stats.ious is not None
        stats.ious.append(score)

    stats.false_positives = len(preds) - len(used_p)
    stats.false_negatives = len(gts) - len(used_g)
    return stats
