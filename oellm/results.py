"""Result collection, metric resolution, and structured output for evaluation outputs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pandas as pd
import yaml

from oellm.constants import METRIC_FALLBACK_KEYS
from oellm.utils import _setup_logging


def _resolve_metric(
    task_name: str, result_dict: dict, task_metrics: dict
) -> tuple[float | None, str | None]:
    """Return (value, metric_name) for task_name from result_dict."""

    # Normalise lmms-eval task-scoped metric keys so lm-eval and lmms-eval
    # output is handled identically.  lmms-eval writes keys like
    # "vqav2/vqa_score,none"; strip the "task_name/" prefix so the lookup
    # below sees "vqa_score,none" regardless of engine.  Keys without "/"
    # (lm-eval format) are passed through unchanged.
    result_dict = {
        (k.split("/", 1)[1] if "/" in k else k): v for k, v in result_dict.items()
    }

    def _first_numeric(d: dict, *candidates: str) -> tuple[float | None, str | None]:
        for c in candidates:
            if c in d and isinstance(d[c], (int, float)):
                return float(d[c]), c
        return None, None

    def _first_matching_prefix(d: dict, prefix: str) -> tuple[float | None, str | None]:
        for k, v in d.items():
            if (k == prefix or k.startswith(prefix + ",")) and isinstance(
                v, (int, float)
            ):
                return float(v), k
        return None, None

    preferred = task_metrics.get(task_name)
    if preferred is not None:
        val, key = _first_numeric(result_dict, f"{preferred},none", preferred)
        if val is not None:
            return val, key
        val, key = _first_matching_prefix(result_dict, preferred)
        return val, key

    for metric in METRIC_FALLBACK_KEYS:
        val, key = _first_numeric(result_dict, metric)
        if val is not None:
            return val, key
        val, key = _first_matching_prefix(result_dict, metric.split(",")[0])
        if val is not None:
            return val, key

    # Last resort: pick the first numeric non-stderr value (catches lmms-eval
    # benchmarks with non-standard metric names like mme_cognition_score)
    for k, v in result_dict.items():
        if (
            isinstance(v, (int, float))
            and "stderr" not in k
            and k not in ("alias", " ", "")
        ):
            return float(v), k
    return None, None


def _infer_global_n_shot(n_shot_data: dict) -> int | None:
    """Infer a global n_shot if exactly one unique value exists."""
    try:
        candidate_values = []
        for _v in n_shot_data.values():
            if isinstance(_v, (int | float)):
                candidate_values.append(int(_v))
            elif isinstance(_v, str) and _v.isdigit():
                candidate_values.append(int(_v))
        unique_values = set(candidate_values)
        if len(unique_values) == 1:
            return next(iter(unique_values))
    except Exception:
        pass
    return None


def _resolve_n_shot(
    task_name: str,
    n_shot_data: dict,
    group_subtasks_map: dict,
    group_aggregate_names: set,
    global_n_shot: int | None,
) -> int | str:
    """Resolve n_shot for a task, with fallbacks for groups and MMLU."""
    n_shot = n_shot_data.get(task_name, "unknown")

    # If this is a group aggregate and n_shot is missing, derive from any subtask
    if task_name in group_aggregate_names and n_shot == "unknown":
        for subtask_name in group_subtasks_map.get(task_name, []):
            if subtask_name in n_shot_data:
                n_shot = n_shot_data[subtask_name]
                break
    if n_shot == "unknown" and global_n_shot is not None:
        n_shot = global_n_shot

    # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
    if task_name == "mmlu" and n_shot == "unknown":
        for key, value in n_shot_data.items():
            if key.startswith("mmlu_"):
                n_shot = value
                break
        if n_shot == "unknown" and global_n_shot is not None:
            n_shot = global_n_shot

    # Special handling for Global MMLU aggregates - get n_shot from subtasks
    if task_name.startswith("global_mmlu_") and n_shot == "unknown":
        prefix = f"{task_name}_"
        for key, value in n_shot_data.items():
            if key.startswith(prefix):
                n_shot = value
                break
        if n_shot == "unknown" and global_n_shot is not None:
            n_shot = global_n_shot

    return n_shot


def _load_task_metrics() -> dict:
    """Load task_metrics from core YAML and all contrib suites."""
    task_groups_yaml = files("oellm.resources") / "task-groups.yaml"
    with open(str(task_groups_yaml)) as _f:
        _tg_cfg = yaml.safe_load(_f)
    task_metrics = _tg_cfg.get("task_metrics", {})

    from oellm.registry import (
        get_all_task_groups as _contrib_task_groups,
    )

    task_metrics.update(_contrib_task_groups().get("task_metrics", {}))
    return task_metrics


def collect_results(
    results_dir: str,
    output_csv: str = "eval_results.csv",
    *,
    check: bool = False,
    verbose: bool = False,
) -> None:
    """
    Collect evaluation results from JSON files and export to CSV.

    Args:
        results_dir: Path to the directory containing result JSON files
        output_csv: Output CSV filename (default: eval_results.csv)
        check: Check for missing evaluations and create a missing jobs CSV
        verbose: Enable verbose logging
    """
    _setup_logging(verbose)

    task_metrics = _load_task_metrics()

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # lm-eval writes flat JSON files: results/<hex>.json
    # lmms-eval writes nested dirs:  results/<hex>.json/<model>/<ts>_results.json
    # rglob("*.json") + is_file() finds both without breaking backward compat.
    search_root = (
        (results_path / "results")
        if (results_path / "results").is_dir()
        else results_path
    )
    json_files = [p for p in search_root.rglob("*.json") if p.is_file()]

    if not json_files:
        logging.warning(f"No JSON files found in {results_dir}")
        if not check:
            return

    logging.info(f"Found {len(json_files)} result files")

    # If check mode, also load the jobs.csv to compare
    if check:
        jobs_csv_path = results_path / "jobs.csv"
        if not jobs_csv_path.exists():
            logging.warning(f"No jobs.csv found in {results_dir}, cannot perform check")
            check = False
        else:
            jobs_df = pd.read_csv(jobs_csv_path)
            logging.info(f"Found {len(jobs_df)} scheduled jobs in jobs.csv")

    rows = []
    completed_jobs = set()

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # lmms-eval sets model_name to the adapter type (e.g. "llava_hf"),
        # not the checkpoint path; the actual path is in model_name_or_path.
        model_name = data.get("model_name_or_path") or data.get("model_name", "unknown")

        results = data.get("results", {})
        n_shot_data = data.get("n-shot", {})

        # lmms-eval has no "n-shot" dict; fall back to per-task config "num_fewshot"
        if not n_shot_data:
            for _task, _cfg in data.get("configs", {}).items():
                if isinstance(_cfg, dict):
                    shot = _cfg.get("num_fewshot")
                    if shot is not None:
                        n_shot_data[_task] = shot

        global_n_shot = _infer_global_n_shot(n_shot_data)

        # Aggregate groups (lm-eval harness)
        groups_map = data.get("groups", {})
        group_subtasks_map = data.get("group_subtasks", {})
        group_aggregate_names = set(groups_map.keys()) | set(group_subtasks_map.keys())
        group_subtask_names: set[str] = set()
        for _agg, _subs in group_subtasks_map.items():
            for _s in _subs:
                group_subtask_names.add(_s)

        # Prefer only the first aggregate metric from groups (simplified)
        if groups_map:
            group_name, group_results = next(iter(groups_map.items()))
            n_shot = n_shot_data.get(group_name, "unknown")
            if n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(group_name, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot
            performance, metric_name = _resolve_metric(
                group_name, group_results, task_metrics
            )
            if performance is not None:
                if check:
                    completed_jobs.add((model_name, group_name, n_shot))
                rows.append(
                    {
                        "model_name": model_name,
                        "task": group_name,
                        "n_shot": n_shot,
                        "performance": performance,
                        "metric_name": metric_name if metric_name is not None else "",
                    }
                )
                # Skip per-task iteration when groups are present
                continue

        for task_name, task_results in results.items():
            # Skip entries already added from groups
            if groups_map and task_name in group_aggregate_names:
                continue
            # Skip any lm-eval group subtasks; keep only aggregates
            if task_name in group_subtask_names:
                continue

            # Skip MMLU subtasks - only keep the aggregate score
            if task_name.startswith("mmlu_") and task_name != "mmlu":
                continue

            # Skip Global MMLU subtasks - keep only aggregates like global_mmlu_full_pt
            if task_name.startswith("global_mmlu_") and task_name.count("_") >= 4:
                continue

            n_shot = _resolve_n_shot(
                task_name,
                n_shot_data,
                group_subtasks_map,
                group_aggregate_names,
                global_n_shot,
            )

            # Skip lmms-eval parent task placeholders (no numeric metrics, just alias)
            if set(task_results.keys()) <= {"alias", " ", ""}:
                continue

            performance, metric_name = _resolve_metric(
                task_name, task_results, task_metrics
            )

            if performance is not None:
                if check:
                    completed_jobs.add((model_name, task_name, n_shot))

                rows.append(
                    {
                        "model_name": model_name,
                        "task": task_name,
                        "n_shot": n_shot,
                        "performance": performance,
                        "metric_name": metric_name if metric_name is not None else "",
                    }
                )
            else:
                # Log missing metrics — for lmms-eval tasks this often means
                # llm_as_judge_eval is null (no judge LLM configured) or the
                # metric key is not yet listed in task_metrics in task-groups.yaml
                logging.warning(
                    f"No numeric metric for '{task_name}' in {json_file.name} "
                    f"— value may be null (LLM judge not configured?) or metric key missing from task_metrics"
                )

    if not rows and not check:
        logging.warning("No results extracted from JSON files")
        return

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        logging.info(f"Extracted {len(df)} evaluation results")

        if verbose:
            logging.info("Summary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    if check:
        logging.info("=== Evaluation Status Check ===")

        missing_jobs = []

        for _, job in jobs_df.iterrows():
            job_tuple = (job["model_path"], job["task_path"], job["n_shot"])

            is_completed = False

            if job_tuple in completed_jobs:
                is_completed = True
            else:
                for completed_job in completed_jobs:
                    completed_model, completed_task, completed_n_shot = completed_job

                    if (
                        job["n_shot"] == completed_n_shot
                        and job["task_path"] == completed_task
                        and (
                            str(job["model_path"]).endswith(completed_model)
                            or completed_model in str(job["model_path"])
                        )
                    ):
                        is_completed = True
                        break

            if not is_completed:
                missing_jobs.append(job)

        completed_count = len(jobs_df) - len(missing_jobs)

        logging.info(f"Total scheduled jobs: {len(jobs_df)}")
        logging.info(f"Completed jobs: {completed_count}")
        logging.info(f"Missing jobs: {len(missing_jobs)}")

        if len(missing_jobs) > 0:
            missing_df = pd.DataFrame(missing_jobs)
            missing_csv = output_csv.replace(".csv", "_missing.csv")
            missing_df.to_csv(missing_csv, index=False)
            logging.info(f"Missing jobs saved to: {missing_csv}")
            logging.info(
                f"You can run these with: oellm schedule-eval --eval_csv_path {missing_csv}"
            )

            if verbose and len(missing_jobs) > 0:
                logging.info("Example missing jobs:")
                for _i, (_, job) in enumerate(missing_df.head(5).iterrows()):
                    logging.info(
                        f"  - {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                    )
                if len(missing_jobs) > 5:
                    logging.info(f"  ... and {len(missing_jobs) - 5} more")


# ---------------------------------------------------------------------------
# Structured output: versioned JSON and Markdown report
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


def write_results_json(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """Write evaluation results as a versioned JSON file.

    The schema is::

        {
            "version": "1.0",
            "generated_at": "2026-04-02T12:00:00+00:00",
            "results": [
                {"model": ..., "task": ..., "n_shot": ..., "metric": ..., "performance": ...}
            ]
        }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        results.append(
            {
                "model": row.get("model_name", ""),
                "task": row.get("task", ""),
                "n_shot": row.get("n_shot", 0),
                "metric": row.get("metric_name", ""),
                "performance": row.get("performance", 0.0),
            }
        )

    envelope = {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }

    output_path.write_text(json.dumps(envelope, indent=2))


def write_results_markdown(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """Write evaluation results as a Markdown table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Model | Task | N-shot | Metric | Performance |",
        "|-------|------|--------|--------|-------------|",
    ]
    for row in rows:
        model = row.get("model_name", "")
        task = row.get("task", "")
        n_shot = row.get("n_shot", 0)
        metric = row.get("metric_name", "")
        perf = row.get("performance", 0.0)
        lines.append(f"| {model} | {task} | {n_shot} | {metric} | {perf:.4f} |")

    output_path.write_text("\n".join(lines) + "\n")
