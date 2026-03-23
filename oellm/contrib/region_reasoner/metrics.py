"""Region-grounding benchmark metrics (model-agnostic).

:class:`BaseMetric` subclasses that compute bounding-box IoU metrics for the
region-grounding benchmark.  These are the **single source of truth** used by
``suite._aggregate_shards()`` to score any model's predictions — not just
RegionReasoner.  Any model that outputs ``predicted_bbox`` / ``gt_bbox`` pairs
in ``[x1, y1, x2, y2]`` format can be evaluated with these metrics.

Bbox format
-----------
Each bounding box is a JSON-serialised list ``"[x1, y1, x2, y2]"`` where
coordinates are in absolute pixels (or normalised — the formulas are
identical as long as predictions and references use the same system).

Inputs to ``compute()`` are ``list[str]`` (one JSON string per sample) to
comply with the ``BaseMetric`` signature.  Empty strings or ``"null"`` are
treated as a missed detection (IoU = 0).

Metrics
-------
- **GIoU**: mean of per-sample intersection-over-union values.
- **CIoU**: cumulative IoU — sum of all intersections / sum of all unions.
- **BboxAP**: fraction of samples where IoU > 0.5 (binarised AP).
- **PassRate**: fraction of samples where IoU > *threshold* (configurable).

All return values are in [0, 1] (higher is better).
"""

from __future__ import annotations  # noqa: I001

import json

from oellm.core.base_metric import BaseMetric


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two axis-aligned bounding boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]``
        box_b: ``[x1, y1, x2, y2]``

    Returns:
        IoU in [0, 1].  Returns 0.0 for degenerate boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def _parse_box(s: str) -> list[float] | None:
    """Parse a JSON-serialised bbox string.  Returns None on failure."""
    if not s or s.strip() in ("null", "none", ""):
        return None
    try:
        val = json.loads(s)
        if isinstance(val, list) and len(val) == 4:
            return [float(v) for v in val]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def _compute_ious(predictions: list[str], references: list[str]) -> list[float]:
    """Return per-sample IoU values (0.0 for unparseable inputs)."""
    ious = []
    for pred_s, ref_s in zip(predictions, references, strict=True):
        pred = _parse_box(pred_s)
        ref = _parse_box(ref_s)
        if pred is None or ref is None:
            ious.append(0.0)
        else:
            ious.append(_box_iou(pred, ref))
    return ious


# ---------------------------------------------------------------------------
# Public metric classes
# ---------------------------------------------------------------------------


class GIoU(BaseMetric):
    """Mean per-sample IoU (gIoU as reported in the RegionReasoner paper).

    Formula: ``mean(intersection_i / union_i)`` over all samples.
    """

    @property
    def name(self) -> str:
        return "gIoU"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0
        ious = _compute_ious(predictions, references)
        return sum(ious) / len(ious)


class CIoU(BaseMetric):
    """Cumulative IoU (cIoU as reported in the RegionReasoner paper).

    Formula: ``sum(all intersections) / sum(all unions)``.
    Different from GIoU — large objects contribute more.
    """

    @property
    def name(self) -> str:
        return "cIoU"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        total_intersection = 0.0
        total_union = 0.0
        for pred_s, ref_s in zip(predictions, references, strict=True):
            pred = _parse_box(pred_s)
            ref = _parse_box(ref_s)
            if pred is None or ref is None:
                # Missed detection: union = area of reference, intersection = 0
                if ref is not None:
                    ax1, ay1, ax2, ay2 = ref
                    total_union += max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                continue
            ax1, ay1, ax2, ay2 = pred
            bx1, by1, bx2, by2 = ref
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            intersection = inter_w * inter_h
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - intersection
            total_intersection += intersection
            total_union += union

        if total_union <= 0.0:
            return 0.0
        return total_intersection / total_union


class BboxAP(BaseMetric):
    """Binarised bounding-box AP at IoU threshold 0.5.

    Formula: ``mean(IoU_i > 0.5)`` — fraction of samples where the predicted
    bbox overlaps the reference by at least 50%.
    """

    @property
    def name(self) -> str:
        return "bbox_AP"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0
        ious = _compute_ious(predictions, references)
        return sum(iou > 0.5 for iou in ious) / len(ious)


class PassRate(BaseMetric):
    """Fraction of samples where IoU exceeds a configurable threshold.

    Args:
        threshold: IoU threshold in (0, 1).  Common values: 0.3, 0.5, 0.7, 0.9.

    Example::

        PassRate(0.3).name   # "pass_rate_0.3"
        PassRate(0.5).compute(preds, refs)
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self._threshold = threshold

    @property
    def name(self) -> str:
        # Format cleanly: 0.3 → "pass_rate_0.3", not "pass_rate_0.30000..."
        t = self._threshold
        label = f"{t:.10g}"
        return f"pass_rate_{label}"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0
        ious = _compute_ious(predictions, references)
        return sum(iou > self._threshold for iou in ious) / len(ious)
