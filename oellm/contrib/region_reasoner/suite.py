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

``REGION_REASONER_TEST_JSON``  *(optional)*
    Absolute path to the test JSON file (``refcocog_multi_turn.json``).
    If not set, the file is downloaded automatically from
    ``lmsdss/regionreasoner_test_data`` on the HF Hub before inference.

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
]

from oellm.contrib.region_reasoner.task import RegionReasonerTask  # noqa: E402

TASK_GROUPS: dict = RegionReasonerTask.to_task_groups_dict()

# ---------------------------------------------------------------------------
# Plugin protocol: optional — model-flag detection
# ---------------------------------------------------------------------------


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to RegionReasonerModelAdapter.to_contrib_flags()."""
    from oellm.contrib.region_reasoner.adapter import RegionReasonerModelAdapter

    return RegionReasonerModelAdapter(model_path).to_contrib_flags()


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
    num_gpus = int(env.get("REGION_REASONER_NUM_GPUS", "4"))
    num_parts = int(env.get("REGION_REASONER_NUM_PARTS", str(num_gpus)))
    model_type = model_flags or "vision_reasoner"

    if not rr_dir:
        raise RuntimeError(
            "REGION_REASONER_DIR must be set. Add it to clusters.yaml for this cluster."
        )

    test_json = env.get("REGION_REASONER_TEST_JSON")
    if not test_json:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id="lmsdss/regionreasoner_test_data",
            repo_type="dataset",
            allow_patterns=["raw/refcocog_multi_turn.json"],
            cache_dir=Path(env["HF_HOME"]) / "hub" if "HF_HOME" in env else None,
        )
        test_json = str(Path(local_dir) / "raw" / "refcocog_multi_turn.json")

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
                "--model",
                model_type,
                "--test_data_path",
                test_json,
                "--output_path",
                tmp_dir,  # script writes <tmp_dir>/output_{idx}.json
                "--vis_output_path",
                str(Path(tmp_dir) / f"vis_{idx}"),
                "--idx",
                str(idx),
                "--num_parts",
                str(num_parts),
                "--batch_size",
                "2",
                "--task_router_model_path",
                "Ricky06662/TaskRouter-1.5B",
            ]
            logger.info("Starting shard %d/%d: %s", idx + 1, num_gpus, " ".join(cmd))
            proc = subprocess.Popen(cmd, env=shard_env, cwd=str(Path(test_json).parent))
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
    """Read per-shard output files and compute all metrics.

    The upstream ``evaluation_multi_segmentation.py`` script writes one entry
    per conversational turn::

        {"intersection": <int>, "union": <int>, "bbox_iou": <float>, ...}

    where ``intersection`` / ``union`` are mask pixel counts and ``bbox_iou``
    is the pre-computed bbox IoU.

    Returns a flat dict of ``{metric_name: value}`` for all seven metrics.
    """
    shard_files = sorted(Path(shard_dir).glob("output_*.json"))
    if not shard_files:
        raise RuntimeError(
            f"No shard output files found in {shard_dir!r}. "
            "The inference script may have failed silently."
        )

    intersections: list[float] = []
    unions: list[float] = []
    bbox_ious: list[float] = []

    for shard_file in shard_files:
        with open(shard_file) as f:
            shard_data = json.load(f)
        for sample in shard_data:
            intersections.append(float(sample.get("intersection", 0)))
            unions.append(float(sample.get("union", 0)))
            bbox_ious.append(float(sample.get("bbox_iou", 0.0)))

    n = len(bbox_ious)
    logger.info("Aggregating %d samples from %d shards", n, len(shard_files))

    if n == 0:
        return dict.fromkeys(
            (
                "gIoU",
                "cIoU",
                "bbox_AP",
                "pass_rate_0.3",
                "pass_rate_0.5",
                "pass_rate_0.7",
                "pass_rate_0.9",
            ),
            0.0,
        )

    giou = (
        sum(i / u if u > 0 else 0.0 for i, u in zip(intersections, unions, strict=True))
        / n
    )
    total_u = sum(unions)
    ciou = sum(intersections) / total_u if total_u > 0 else 0.0

    metrics = {
        "gIoU": giou,
        "cIoU": ciou,
        "bbox_AP": sum(iou > 0.5 for iou in bbox_ious) / n,
        "pass_rate_0.3": sum(iou > 0.3 for iou in bbox_ious) / n,
        "pass_rate_0.5": sum(iou > 0.5 for iou in bbox_ious) / n,
        "pass_rate_0.7": sum(iou > 0.7 for iou in bbox_ious) / n,
        "pass_rate_0.9": sum(iou > 0.9 for iou in bbox_ious) / n,
    }
    for name, val in metrics.items():
        logger.debug("%s = %.4f", name, val)
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
