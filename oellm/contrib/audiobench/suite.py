"""AudioBench contrib suite — plugin protocol implementation.

Implements the :mod:`oellm.registry` plugin protocol for the AudioBench
benchmark (AudioLLMs/AudioBench, arXiv 2406.16020).  AudioBench is **not** a
pip-installable library — it is a script harness.  We invoke its entry point
via ``python src/main_evaluate.py`` as a subprocess, from a clone pointed at
by the ``$AUDIOBENCH_DIR`` environment variable (configured in
``clusters.yaml``).  This mirrors the precedent set by ``regiondial_bench``.

Cluster setup
-------------
The following environment variables must be set in ``clusters.yaml`` (or the
cluster's module/profile system) before using any ``audio-audiobench-*``
task group:

``AUDIOBENCH_DIR``
    Absolute path to a local clone of
    https://github.com/AudioLLMs/AudioBench.  The entry point
    ``src/main_evaluate.py`` must be present and the repo's own Python
    dependencies must be installed in the active environment.

Phase 2 (judge-dependent tasks) will additionally require:

``AUDIOBENCH_JUDGE_URL`` / ``AUDIOBENCH_JUDGE_MODEL``
    OpenAI-compatible URL and model name for the judge server (typically a
    vLLM deployment of ``meta-llama/Meta-Llama-3-70B-Instruct-AWQ``).  Not
    needed for Phase-1 judge-free tasks shipped today.

Output format
-------------
:func:`run` writes a lmms-eval-compatible JSON file to *output_path* so
that :func:`oellm.main.collect_results` can parse it without modification::

    {
      "model_name_or_path": "<model_path>",
      "results": {
        "audiobench_librispeech_test_clean": {
          "wer": 0.047
        }
      },
      "configs": {
        "audiobench_librispeech_test_clean": {"num_fewshot": 0}
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from oellm.contrib.audiobench.task import (
    AUDIOBENCH_TASKS,
    SUITE_NAME,
    AudioBenchTaskSpec,
    get_task_spec,
)

logger = logging.getLogger(__name__)

CLUSTER_ENV_VARS = ["AUDIOBENCH_DIR"]

# Mapping family → (group_name, human description).
_FAMILY_GROUPS = {
    "asr": (
        "audio-audiobench-asr",
        "AudioBench ASR tasks (WER).  Covers AudioBench-scored LibriSpeech, "
        "Common Voice 15 EN, GigaSpeech, People's Speech, TED-LIUM 3 — dual "
        "with our lmms-eval versions for paper-comparable numbers — plus new "
        "tasks not in lmms-eval: earnings21/22, TED-LIUM 3 long-form, "
        "AISHELL Mandarin, GigaSpeech2 (th/id/vi), SEAME code-switch.",
    ),
    "st": (
        "audio-audiobench-st",
        "AudioBench speech-translation tasks (BLEU).  CoVoST2 covering "
        "en↔id, en↔ta, zh→en, ta→en (new), plus dual-registered en→zh.",
    ),
    "reasoning": (
        "audio-audiobench-reasoning",
        "AudioBench spoken reasoning / captioning.  Spoken-MQA digit + "
        "reasoning splits (accuracy), MMAU-mini (string_match), "
        "AudioCaps (METEOR).",
    ),
}

_TOP_LEVEL_GROUP = "audio-audiobench"
_TOP_LEVEL_DESC = (
    "AudioBench Phase-1 suite (judge-free).  Runs all 27 AudioBench tasks "
    "that do not require an LLM judge: ASR (WER), speech translation (BLEU), "
    "spoken reasoning (accuracy/string_match), and AudioCaps captioning "
    "(METEOR).  Phase 2 (judge-dependent tasks) will extend this group once "
    "the judge service is configured."
)


def _build_task_groups() -> dict:
    """Assemble the :data:`TASK_GROUPS` dict from :data:`AUDIOBENCH_TASKS`.

    One top-level ``audio-audiobench`` group containing all 27 leaves, plus
    three sub-groups keyed by family (``-asr`` / ``-st`` / ``-reasoning``).
    All groups are zero-shot by design — AudioBench tasks do not support
    in-context examples.
    """
    task_metrics: dict[str, str] = {t.name: t.metric for t in AUDIOBENCH_TASKS}

    def _task_entry(t: AudioBenchTaskSpec) -> dict:
        entry: dict = {"task": t.name, "dataset": t.hf_repo}
        # ``data_dir``-style subsetting: we deliberately do NOT set ``subset``
        # in the YAML entry.  The reason is that ``load_dataset(name=...)``
        # used by ``_pre_download_datasets_from_specs`` treats ``subset`` as a
        # config name, not a ``data_dir`` — and for gigaspeech2/spoken-mqa the
        # upstream distinction is a data_dir, not a config.  Since the group
        # name starts with "audio-", ``_collect_dataset_specs`` auto-sets
        # ``needs_snapshot_download=True`` which downloads the whole repo,
        # so AudioBench can read the right data_dir at runtime.  This also
        # means multiple tasks sharing one HF repo dedupe to a single spec.
        return entry

    groups: dict[str, dict] = {}

    # Sub-groups per family.
    tasks_by_family: dict[str, list[AudioBenchTaskSpec]] = {
        "asr": [],
        "st": [],
        "reasoning": [],
    }
    for t in AUDIOBENCH_TASKS:
        tasks_by_family[t.family].append(t)

    for family, (group_name, desc) in _FAMILY_GROUPS.items():
        entries = tasks_by_family[family]
        if not entries:
            continue
        groups[group_name] = {
            "suite": SUITE_NAME,
            "n_shots": [0],
            "description": desc,
            "tasks": [_task_entry(t) for t in entries],
        }

    # Top-level group — union of everything.
    groups[_TOP_LEVEL_GROUP] = {
        "suite": SUITE_NAME,
        "n_shots": [0],
        "description": _TOP_LEVEL_DESC,
        "tasks": [_task_entry(t) for t in AUDIOBENCH_TASKS],
    }

    return {"task_metrics": task_metrics, "task_groups": groups}


TASK_GROUPS: dict = _build_task_groups()


# ---------------------------------------------------------------------------
# Model-flag detection.
# ---------------------------------------------------------------------------


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to :class:`AudioBenchModelAdapter`.

    Called by :class:`oellm.runner.EvalRunner.resolve_suite` to append the
    AudioBench model-family key to ``eval_suite`` as
    ``audiobench:<family>``.
    """
    from oellm.contrib.audiobench.adapter import AudioBenchModelAdapter

    return AudioBenchModelAdapter(model_path).to_contrib_flags()


# ---------------------------------------------------------------------------
# Runtime — subprocess into AudioBench's src/main_evaluate.py.
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
    """Execute one AudioBench task and write lmms-eval-shaped JSON.

    Args:
        model_path: HF repo ID or local path of the model under evaluation.
        task: Canonical task name (must start with ``audiobench_``).
        n_shot: Always 0 for AudioBench — recorded in the output ``configs``
            block for downstream compatibility.
        output_path: Destination for the lmms-eval-compatible result JSON.
        model_flags: AudioBench ``--model`` key (e.g. ``"qwen2_audio"``);
            produced by :func:`detect_model_flags`.  Falls back to
            ``"generic"`` if not supplied.
        env: Environment dict passed to the subprocess.  Must contain
            ``AUDIOBENCH_DIR`` (validated by dispatch.py before ``run`` is
            called, but we re-check for safety).

    Raises:
        RuntimeError: if AudioBench returns non-zero or produces no output.
        KeyError: if *task* is not in the registry.
    """
    ab_dir = env.get("AUDIOBENCH_DIR")
    if not ab_dir:
        raise RuntimeError(
            "AUDIOBENCH_DIR must be set.  Add it to clusters.yaml — "
            "it should point at a local clone of "
            "https://github.com/AudioLLMs/AudioBench."
        )

    entrypoint = Path(ab_dir) / "src" / "main_evaluate.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"AudioBench entry point not found: {entrypoint}\n"
            f"Check that AUDIOBENCH_DIR={ab_dir!r} points at a valid "
            "AudioBench clone."
        )

    spec = get_task_spec(task)
    model_key = model_flags or "generic"

    # AudioBench writes outputs under a run-specific log directory; we set
    # it to our output_path's parent so we can recover the raw result.
    run_dir = output_path.parent / f"audiobench_{output_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "src/main_evaluate.py",
        "--dataset",
        spec.upstream_name,
        "--model",
        model_key,
        "--model_name",
        model_path,
        "--metrics",
        spec.upstream_metric,
        "--log_dir",
        str(run_dir),
    ]
    if spec.data_dir:
        cmd.extend(["--data_dir", spec.data_dir])

    # Forward LIMIT (set by template.sbatch) as AudioBench's
    # --number_of_samples when present.  "-1" means no limit in AudioBench.
    limit = env.get("LIMIT", "").strip()
    if limit:
        cmd.extend(["--number_of_samples", str(limit)])

    logger.info("AudioBench cmd: %s (cwd=%s)", " ".join(cmd), ab_dir)
    completed = subprocess.run(
        cmd,
        cwd=ab_dir,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"AudioBench exited with code {completed.returncode} for "
            f"task={task!r} model={model_path!r}"
        )

    metrics = _extract_metrics(run_dir, spec)
    _write_lmms_shaped_json(
        output_path=output_path,
        model_path=model_path,
        task_name=task,
        n_shot=n_shot,
        metrics=metrics,
    )
    logger.info("Results written to %s", output_path)


def _extract_metrics(run_dir: Path, spec: AudioBenchTaskSpec) -> dict[str, float]:
    """Find AudioBench's per-task score JSON inside *run_dir* and read it.

    AudioBench writes one JSON file per task under its ``--log_dir`` with
    the score under a key matching ``--metrics``.  We search recursively
    for any ``*.json`` and pick the first one whose body contains the
    expected metric key.  This is intentionally lenient because upstream
    log-layout has changed across releases.

    Raises:
        RuntimeError: if no matching result file is found.
    """
    candidates = sorted(run_dir.rglob("*.json"))
    if not candidates:
        raise RuntimeError(
            f"AudioBench produced no result JSON under {run_dir}.  "
            "Check stdout/stderr for crashes."
        )

    target_key = spec.upstream_metric
    for path in candidates:
        try:
            with open(path) as f:
                body = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        value = _find_metric(body, target_key)
        if value is not None:
            # Emit the metric under OUR canonical key (spec.metric) so the
            # lmms-eval-style ``task/metric,none`` stripping in
            # collect_results() resolves to what's in task_metrics.yaml.
            return {spec.metric: float(value)}

    raise RuntimeError(
        f"Could not locate metric {target_key!r} in any of "
        f"{len(candidates)} AudioBench result JSON(s) under {run_dir}"
    )


def _find_metric(body: object, key: str) -> float | None:
    """Recursive search for a numeric value keyed by *key* anywhere in *body*.

    AudioBench's per-task JSON has nested structure that has drifted across
    releases (sometimes ``{"wer": 0.04}``, sometimes
    ``{"metrics": {"wer": {"score": 0.04}}}``).  We tolerate either form.
    """
    if isinstance(body, dict):
        if key in body:
            candidate = body[key]
            if isinstance(candidate, int | float):
                return float(candidate)
            if isinstance(candidate, dict) and "score" in candidate:
                score = candidate["score"]
                if isinstance(score, int | float):
                    return float(score)
        for v in body.values():
            found = _find_metric(v, key)
            if found is not None:
                return found
    elif isinstance(body, list):
        for item in body:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _write_lmms_shaped_json(
    *,
    output_path: Path,
    model_path: str,
    task_name: str,
    n_shot: int,
    metrics: dict[str, float],
) -> None:
    """Write a lmms-eval-compatible JSON at *output_path*.

    :func:`oellm.main.collect_results` reads this shape directly; the
    ``_resolve_metric`` fallback chain picks up our ``task_metrics``
    mapping to extract the primary value.
    """
    payload = {
        "model_name_or_path": model_path,
        "results": {task_name: metrics},
        "configs": {task_name: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# parse_results — invoked by collect_results to recognise our output files.
# ---------------------------------------------------------------------------


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Recognise a JSON dict produced by :func:`run`.

    Detection heuristic: the ``results`` dict contains at least one key
    that starts with ``"audiobench_"``.  Returns the tuple expected by
    :func:`oellm.main.collect_results`:

        ``(model_id, task_name, n_shot, {metric: value})``

    Returns ``None`` for JSON blobs that don't belong to this suite.
    """
    results = data.get("results", {})
    if not isinstance(results, dict):
        return None
    for task_name, task_results in results.items():
        if not isinstance(task_name, str) or not task_name.startswith("audiobench_"):
            continue
        if not isinstance(task_results, dict):
            continue
        model_id = data.get("model_name_or_path") or data.get("model_name") or "unknown"
        n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
        # Coerce everything that can be float; leave non-numeric alone so
        # _resolve_metric can still see them.
        coerced: dict[str, float] = {}
        for k, v in task_results.items():
            if isinstance(v, int | float):
                coerced[k] = float(v)
        return model_id, task_name, int(n_shot), coerced
    return None


# Re-exports used by the test suite.
__all__ = [
    "CLUSTER_ENV_VARS",
    "SUITE_NAME",
    "TASK_GROUPS",
    "detect_model_flags",
    "parse_results",
    "run",
]

# Silence unused-import lint (the symbol is exported for consumer reuse).
_ = os
