"""RegionReasoner contrib suite — plugin protocol implementation.

This module follows the plugin protocol defined in ``oellm/registry.py``.
It is the reference implementation for custom benchmark integration.

Cluster setup
-------------
The following environment variables must be set in ``clusters.yaml`` (or the
cluster's module/profile system) before using the ``region-reasoner`` task group:

``REGION_REASONER_DIR``
    Absolute path to a local clone of the RegionReasoner repository
    (https://github.com/lmsdss/RegionReasoner).  The eval scripts
    ``test/evaluation/evaluation_multi_segmentation.py`` must be present.

``REGION_REASONER_TEST_JSON``
    Absolute path to the prepared test JSON file (e.g.
    ``refcocog_multi_turn.json``).  See the RegionReasoner repo for data
    preparation instructions.

``REGION_REASONER_NUM_GPUS``  *(optional, default: 4)*
    Number of GPU shards to use for parallel inference.  Should match the
    number of GPUs available on the compute node.

Output format
-------------
``run()`` writes a **lmms-eval-compatible JSON** file so that
``oellm.main.collect_results()`` works without modification::

    {
      "model_name_or_path": "<model_path>",
      "results": {
        "regionreasoner_refcocog": {
          "gIoU": 0.42, "cIoU": 0.45, "bbox_AP": 0.38,
          "pass_rate_0.3": 0.71, "pass_rate_0.5": 0.55,
          "pass_rate_0.7": 0.31, "pass_rate_0.9": 0.08
        }
      },
      "configs": {
        "regionreasoner_refcocog": {"num_fewshot": 0}
      }
    }
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin protocol: required constants
# ---------------------------------------------------------------------------

SUITE_NAME = "region_reasoner"

CLUSTER_ENV_VARS = [
    "REGION_REASONER_DIR",
    "REGION_REASONER_TEST_JSON",
]

TASK_GROUPS: dict = {
    "task_metrics": {
        # Primary metric used by collect_results() for the results CSV
        "regionreasoner_refcocog": "gIoU",
    },
    "task_groups": {
        "region-reasoner": {
            "description": (
                "RegionReasoner multi-turn region grounding benchmark (RefCOCOg). "
                "Requires REGION_REASONER_DIR and REGION_REASONER_TEST_JSON on cluster."
            ),
            "suite": "region_reasoner",
            "n_shots": [0],
            "tasks": [
                {
                    "task": "regionreasoner_refcocog",
                    "dataset": "lmsdss/regionreasoner_data",
                }
            ],
        }
    },
}

# ---------------------------------------------------------------------------
# Plugin protocol: optional — model-flag detection
# ---------------------------------------------------------------------------


def detect_model_flags(model_path: str) -> str | None:
    """Map a model path to the ``--model_type`` flag for the eval script.

    The returned string is appended to the ``eval_suite`` column as
    ``region_reasoner:<flags>`` and passed back to :func:`run` as
    *model_flags*.

    ``vision_reasoner`` is the correct value for ``lmsdss/RegionReasoner-7B``.
    ``qwen2`` / ``qwen`` cover evaluating baseline Qwen2.5-VL / Qwen-VL models
    directly on the benchmark.
    """
    name = Path(model_path).name.lower()
    if "regionreasoner" in name or "region_reasoner" in name:
        return "vision_reasoner"
    if "qwen2" in name:
        return "qwen2"
    if "qwen" in name:
        return "qwen"
    # Default: assume the model is a RegionReasoner-style checkpoint
    return "vision_reasoner"


# ---------------------------------------------------------------------------
# Plugin protocol: required — run evaluation
# ---------------------------------------------------------------------------


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Execute the RegionReasoner evaluation and write results to *output_path*.

    This function:

    1. Runs ``evaluation_multi_segmentation.py`` in parallel GPU shards.
    2. Computes gIoU, cIoU, bbox_AP, and pass_rate at four thresholds using
       the :mod:`oellm.contrib.region_reasoner.metrics` implementations.
    3. Writes a lmms-eval-compatible JSON to *output_path*.

    Args:
        model_path: Path or HF repo ID of the model checkpoint.
        task: Task name (e.g. ``"regionreasoner_refcocog"``).
        n_shot: Number of few-shot examples (always 0 for this benchmark).
        output_path: Where to write the results JSON.
        model_flags: Model type string (e.g. ``"vision_reasoner"``).
        env: Environment variables dict (from ``os.environ``).
    """
    rr_dir = env.get("REGION_REASONER_DIR", "")
    test_json = env.get("REGION_REASONER_TEST_JSON", "")
    num_gpus = int(env.get("REGION_REASONER_NUM_GPUS", "4"))
    model_type = model_flags or "vision_reasoner"

    if not rr_dir or not test_json:
        raise RuntimeError(
            "REGION_REASONER_DIR and REGION_REASONER_TEST_JSON must be set. "
            "Add them to clusters.yaml for this cluster."
        )

    inference_script = (
        Path(rr_dir) / "test" / "evaluation" / "evaluation_multi_segmentation.py"
    )
    if not inference_script.exists():
        raise FileNotFoundError(
            f"RegionReasoner inference script not found: {inference_script}\n"
            f"Check that REGION_REASONER_DIR={rr_dir!r} points to a valid clone."
        )

    with tempfile.TemporaryDirectory(prefix="rr_shards_") as tmp_dir:
        # --- Step 1: parallel GPU shards ---
        procs = []
        for idx in range(num_gpus):
            shard_env = dict(env)
            shard_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            cmd = [
                "python",
                str(inference_script),
                "--model_path",
                model_path,
                "--model_type",
                model_type,
                "--test_json",
                test_json,
                "--output_dir",
                tmp_dir,
                "--idx",
                str(idx),
                "--num_parts",
                str(num_gpus),
                "--batch_size",
                "2",
            ]
            logger.info("Starting shard %d/%d: %s", idx + 1, num_gpus, " ".join(cmd))
            proc = subprocess.Popen(cmd, env=shard_env)
            procs.append(proc)

        for idx, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(
                    f"RegionReasoner inference shard {idx} exited with code {ret}"
                )

        logger.info("All %d shards completed. Computing metrics.", num_gpus)

        # --- Step 2: compute metrics from shard outputs ---
        metrics = _aggregate_shards(tmp_dir)

    # --- Step 3: write lmms-eval-compatible JSON ---
    result_json = {
        "model_name_or_path": model_path,
        "results": {task: metrics},
        "configs": {task: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info("Results written to %s", output_path)


def _aggregate_shards(shard_dir: str) -> dict[str, float]:
    """Read per-shard prediction files and compute all metrics.

    Each shard writes a JSON file containing a list of samples, where each
    sample has ``"pred_bbox"`` and ``"gt_bbox"`` keys (JSON-serialised
    ``[x1, y1, x2, y2]`` lists).

    Returns a flat dict of ``{metric_name: value}`` for all six metrics.
    """
    from oellm.contrib.region_reasoner.metrics import (
        BboxAP,
        CIoU,
        GIoU,
        PassRate,
    )

    # Collect all predictions across shards
    predictions: list[str] = []
    references: list[str] = []

    shard_files = sorted(Path(shard_dir).glob("*.json"))
    if not shard_files:
        raise RuntimeError(
            f"No shard output files found in {shard_dir!r}. "
            "The inference script may have failed silently."
        )

    for shard_file in shard_files:
        with open(shard_file) as f:
            shard_data = json.load(f)

        # Expected format: list of {"pred_bbox": [...], "gt_bbox": [...]}
        # Adjust key names below if the upstream script uses different names.
        for sample in shard_data:
            pred = sample.get("pred_bbox") or sample.get("predicted_bbox")
            ref = sample.get("gt_bbox") or sample.get("ground_truth_bbox")
            predictions.append(json.dumps(pred) if pred is not None else "null")
            references.append(json.dumps(ref) if ref is not None else "null")

    logger.info(
        "Aggregating %d samples from %d shards", len(predictions), len(shard_files)
    )

    metrics: dict[str, float] = {}
    for metric in [
        GIoU(),
        CIoU(),
        BboxAP(),
        PassRate(0.3),
        PassRate(0.5),
        PassRate(0.7),
        PassRate(0.9),
    ]:
        metrics[metric.name] = metric.compute(predictions, references)
        logger.debug("%s = %.4f", metric.name, metrics[metric.name])

    return metrics


# ---------------------------------------------------------------------------
# Plugin protocol: required — parse results JSON
# ---------------------------------------------------------------------------


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Try to parse *data* as a region_reasoner output JSON.

    Returns ``(model_id, task_name, n_shot, {metric: value})`` if the JSON
    matches this suite's format, otherwise ``None``.

    Detection heuristic: the ``results`` dict contains a key that starts with
    ``"regionreasoner_"`` and the value dict contains ``"gIoU"``.
    """
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if task_name.startswith("regionreasoner_") and "gIoU" in task_results:
            model_id = data.get("model_name_or_path") or data.get("model_name", "unknown")
            n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
            return model_id, task_name, int(n_shot), task_results
    return None
