"""DocVQA 2026 contrib suite — plugin protocol implementation.

Replicates the ICDAR 2026 competition benchmark
(https://github.com/VLR-CVC/DocVQA2026): reasoning questions over multi-page
documents in eight domains, scored by the competition's own matcher.

Why a contrib suite rather than an lmms-eval task: the benchmark scores one
question against a whole document (~36 page images), demands the
``FINAL ANSWER:`` marker, and grades with a strict number/unit/date matcher
that falls back to ANLS only for non-numeric ground truths. None of that is
expressible as an lmms-eval task config, and the scorer is the benchmark.

The scorer is vendored byte-for-byte in ``_vendor_eval_utils.py`` and pinned
against upstream by tests/test_docvqa2026_parity.py.

Only the val split is evaluable: the test split's answers are withheld and
graded solely on the RRC platform.

Configuration
-------------
``DOCVQA2026_MAX_PAGES``
    Optional cap on pages per document. Unset means every page, which is what
    the competition scores; a cap makes the benchmark tractable for models that
    cannot hold ~50k image tokens, and is recorded in the results.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "docvqa2026"

# The dataset is staged from the Hub; nothing cluster-local is required.
CLUSTER_ENV_VARS: list[str] = []

from oellm.contrib.docvqa2026.task import DocVQA2026ValTask  # noqa: E402

_TASKS = (DocVQA2026ValTask(),)

_group_kwargs = {"suite": SUITE_NAME, "n_shots": [0]}


def _task_entry(task) -> dict:
    entry: dict = {"task": task.engine_task_name}
    if task.dataset_specs:
        entry["dataset"] = task.dataset_specs[0].repo_id
    return entry


TASK_GROUPS: dict = {
    "task_metrics": {t.engine_task_name: t.primary_metric for t in _TASKS},
    "task_groups": {
        t.task_group_name: {
            **_group_kwargs,
            "description": t.description,
            "tasks": [_task_entry(t)],
        }
        for t in _TASKS
    },
}


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to DocVQA2026Adapter.to_contrib_flags()."""
    from oellm.contrib.docvqa2026.adapter import DocVQA2026Adapter

    return DocVQA2026Adapter(model_path).to_contrib_flags()


def run(
    *,
    model_path: str,
    task: str,
    n_shot: int,
    output_path: Path,
    model_flags: str | None,
    env: dict[str, str],
) -> None:
    """Evaluate *task* and write a lmms-eval-compatible JSON to *output_path*."""
    from oellm.contrib.docvqa2026.datasets import load_val, read_max_pages
    from oellm.contrib.docvqa2026.metrics import aggregate, score_prediction
    from oellm.contrib.docvqa2026.prompts import MASTER_PROMPT
    from oellm.contrib.docvqa2026.runner import (
        generate_answer,
        load_model,
        read_limit,
        resolve_device,
        write_results,
    )

    known = [t.engine_task_name for t in _TASKS]
    if task not in known:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {known}")

    limit = read_limit(env)
    max_pages = read_max_pages(env)
    samples = load_val(limit=limit, max_pages=max_pages)
    logger.info(
        "DocVQA 2026: %d questions over %d documents",
        len(samples),
        len({s.doc_id for s in samples}),
    )

    device = resolve_device()
    model, processor = load_model(model_path, device)

    records = []
    truncated = 0
    for i, sample in enumerate(samples, 1):
        raw = generate_answer(model, processor, sample, MASTER_PROMPT, device)
        correct, extracted, has_marker = score_prediction(raw, sample.answer)
        truncated += sample.pages_truncated
        records.append(
            {
                "question_id": sample.question_id,
                "doc_category": sample.doc_category,
                "correct": correct,
                "extracted": extracted,
                "has_marker": has_marker,
            }
        )
        if i % 10 == 0 or i == len(samples):
            logger.info("scored %d/%d questions", i, len(samples))

    metrics = aggregate(records)
    metrics["max_pages"] = max_pages if max_pages else 0
    metrics["n_truncated_documents"] = truncated
    if not metrics["format_compliance"]:
        logger.warning(
            "no prediction carried the %r marker, so every answer scores wrong "
            "regardless of content — this measures instruction following, not "
            "document reasoning",
            "FINAL ANSWER:",
        )
    if truncated:
        logger.warning(
            "%d/%d questions saw a truncated document (DOCVQA2026_MAX_PAGES=%s); "
            "scores are not comparable with the competition leaderboard",
            truncated,
            len(records),
            max_pages,
        )

    write_results(output_path, model_path, task, n_shot, metrics)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it is this suite's output, else return None."""
    from oellm.contrib.docvqa2026.runner import parse_suite_results

    return parse_suite_results(data, "docvqa2026")
