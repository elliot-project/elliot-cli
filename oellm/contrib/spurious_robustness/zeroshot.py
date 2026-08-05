"""Zero-shot inference primitives shared by the three benchmarks.

These mirror the reference implementation's scoring rules exactly. The two that
are easy to get wrong, because the benchmarks disagree:

* CelebA and UrbanCars score a class as the **maximum** similarity over that
  class's prompts (:func:`class_scores`).
* ImageNet **averages** its 80 templates per class, inside OpenCLIP's
  ``build_zero_shot_classifier``.

Both rules are deliberate and must not be unified.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def encode_texts(model, tokenizer, texts: Sequence[str], device: str) -> torch.Tensor:
    """L2-normalised text embeddings, returned on CPU as ``(len(texts), D)``.

    Encoding runs under ``autocast`` to match the reference implementation.
    This is not a performance detail: reduced-precision accumulation shifts
    embeddings by ~1e-2, which is enough to flip the argmax on borderline
    samples and move worst-group accuracy. Dropping it makes scores diverge
    from the published ones.
    """
    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float().cpu(), dim=-1)


def encode_images(
    model,
    preprocess,
    images: Iterable[Any],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """L2-normalised image embeddings, ``(N, D)`` on CPU.

    Accepts PIL images (how the HF parquet datasets arrive) or filesystem paths
    (how the UrbanCars image tree arrives).
    """
    from PIL import Image

    batch: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        stacked = torch.stack(batch).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            features = model.encode_image(stacked)
        chunks.append(features.float().cpu())
        batch.clear()

    for item in images:
        img = Image.open(item) if isinstance(item, str) else item
        batch.append(preprocess(img.convert("RGB")))
        if len(batch) == batch_size:
            flush()
    flush()

    if not chunks:
        return torch.empty(0)
    return F.normalize(torch.cat(chunks, dim=0), dim=-1)


def class_scores(
    image_features: torch.Tensor, prompt_features: Sequence[torch.Tensor]
) -> np.ndarray:
    """Per-class similarity scores, ``(N, n_classes)``.

    A class described by several prompts scores as the *maximum* similarity over
    its prompts, not the mean: averaging the embeddings of "black hair", "red
    hair" and "bald" yields a centroid that resembles none of them, whereas the
    max keeps each prompt usable as a prototype on its own.
    """
    columns = [(image_features @ pf.T).numpy().max(axis=1) for pf in prompt_features]
    return np.stack(columns, axis=-1)


def predict(scores: np.ndarray) -> np.ndarray:
    """Predicted class index per image."""
    return np.argmax(scores, axis=-1)


def group_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    group_names: Sequence[str],
) -> dict:
    """Overall accuracy, per-group accuracy, and worst-group accuracy.

    ``group_names[i]`` is the subgroup of sample ``i``. Average accuracy is over
    all samples, *not* the mean of the per-group accuracies — the groups are
    heavily imbalanced (CelebA's blonde-male group is under 1% of the split), so
    the two differ substantially and only the former matches the paper.

    Worst-group accuracy is the minimum over the groups that actually occur in
    the data. A group that contributes zero samples cannot have an accuracy and
    is therefore absent from the minimum rather than counted as 0.0 — scoring an
    unobserved group as a total failure would make any subsampled run report a
    worst-group of zero. ``group_counts`` and ``n_groups`` are returned so a run
    that silently covered fewer groups than the benchmark defines is visible in
    the results rather than hidden inside the headline number.
    """
    predictions_arr = np.asarray(predictions)
    labels_arr = np.asarray(labels)
    groups = np.asarray(group_names)

    if labels_arr.size == 0:
        raise ValueError("no samples to score")

    correct = predictions_arr == labels_arr
    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        per_group[name] = float(correct[mask].mean())
        counts[name] = int(mask.sum())

    if not per_group:
        raise ValueError("no groups found")

    worst = min(per_group, key=per_group.__getitem__)
    best = max(per_group, key=per_group.__getitem__)
    avg = float(correct.mean())
    return {
        "avg_accuracy": avg,
        "worst_group_accuracy": per_group[worst],
        "worst_group": worst,
        "best_group_accuracy": per_group[best],
        "best_group": best,
        "accuracy_gap": avg - per_group[worst],
        "group_accuracies": per_group,
        "group_counts": counts,
        "n_groups": len(per_group),
        "total_images": int(labels_arr.size),
    }
