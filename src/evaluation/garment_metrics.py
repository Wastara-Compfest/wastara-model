from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GarmentLevelMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )
        return (
            (self.true_positives + self.true_negatives) / total if total else 0.0
        )

    @property
    def false_positive_rate(self) -> float:
        d = self.false_positives + self.true_negatives
        return self.false_positives / d if d else 0.0


def evaluate_garment_level(
    predictions: list[dict],
    ground_truth: list[dict],
) -> GarmentLevelMetrics:
    """Compare garment-level status.

    Each item: {"track_id": int, "status": "NORMAL"|"DEFECT"}
    Matching is by track_id.
    """
    gt_map = {int(g["track_id"]): g["status"] for g in ground_truth}
    metrics = GarmentLevelMetrics()
    seen = set()
    for pred in predictions:
        tid = int(pred["track_id"])
        seen.add(tid)
        p = pred["status"]
        g = gt_map.get(tid)
        if g is None:
            continue
        if p == "DEFECT" and g == "DEFECT":
            metrics.true_positives += 1
        elif p == "DEFECT" and g == "NORMAL":
            metrics.false_positives += 1
        elif p == "NORMAL" and g == "DEFECT":
            metrics.false_negatives += 1
        elif p == "NORMAL" and g == "NORMAL":
            metrics.true_negatives += 1

    for tid, g in gt_map.items():
        if tid in seen:
            continue
        if g == "DEFECT":
            metrics.false_negatives += 1
        else:
            metrics.true_negatives += 1
    return metrics
