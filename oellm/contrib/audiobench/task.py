"""AudioBench task registry.

Single source of truth for the AudioBench (AudioLLMs/AudioBench, arXiv 2406.16020)
Phase-1 task set.  The registry is consumed by :mod:`oellm.contrib.audiobench.suite`
to auto-generate ``TASK_GROUPS`` and to look up per-task metadata (HF repo,
upstream task name, metric) at dispatch time.

Phase 1 = judge-free tasks only (27 total):

- **20 new** benchmarks not covered by our lmms-eval task groups
  (``earnings{21,22}``, ``gigaspeech2`` {thai, indonesian, vietnamese},
  ``aishell`` ZH ASR, ``seame`` code-switch, covost2 extra language pairs,
  ``spoken-mqa`` reasoning splits, ``mmau_mini``, ``audiocaps`` METEOR).
- **7 dual-registered** duplicates of benchmarks we already run via lmms-eval
  (LibriSpeech test-clean/other, Common Voice 15 EN, GigaSpeech, People's
  Speech, TED-LIUM 3, covost2 en→zh).  These use AudioBench's own scorer
  and normalizer so WP4 can compare numbers against the AudioBench paper.

Naming
------
Every task name is prefixed ``audiobench_`` so the CSV ``task_path`` column
uniquely identifies the scorer and there is no collision with lmms-eval's
``librispeech_test_clean`` etc.  :func:`AudioBenchTaskSpec.upstream_name`
returns the bare name that AudioBench's ``src/main_evaluate.py --dataset``
flag expects.

Phase 2 (judge-dependent tasks) will extend this registry with ~19 more
entries driven by a vLLM Llama-3-70B judge or the OpenAI API; see the
plugin README for the rollout plan.
"""

from __future__ import annotations

from dataclasses import dataclass

SUITE_NAME = "audiobench"
_TASK_NAME_PREFIX = "audiobench_"


@dataclass(frozen=True)
class AudioBenchTaskSpec:
    """Metadata for a single AudioBench task.

    Attributes:
        name: Canonical ``audiobench_*`` task name used in the CSV
            ``task_path`` column and in ``task_metrics`` / ``task_groups``.
        upstream_name: The ``--dataset`` value AudioBench's
            ``src/main_evaluate.py`` expects (e.g. ``"librispeech_test_clean"``).
        hf_repo: HuggingFace dataset repo ID for pre-download
            (e.g. ``"AudioLLMs/librispeech_test_clean"``).
        metric: Primary metric key written to our ``task_metrics`` mapping.
            One of ``wer`` / ``bleu`` / ``accuracy`` / ``string_match`` /
            ``meteor``.
        upstream_metric: The value passed to AudioBench's ``--metrics`` CLI
            flag.  Usually identical to :attr:`metric` but allows divergence
            when AudioBench uses a different key for the same score (e.g.
            ``wer`` vs ``bleu`` match; ``accuracy`` vs upstream ``acc``).
        family: One of ``"asr" | "st" | "reasoning"``.  Controls which
            ``audio-audiobench-*`` sub-group the task lands in.
        data_dir: Optional upstream ``data_dir=...`` selector, used by the
            gigaspeech2 multi-language repo.  Passed to AudioBench via
            ``--data_dir`` (upstream convention).
    """

    name: str
    upstream_name: str
    hf_repo: str
    metric: str
    upstream_metric: str
    family: str
    data_dir: str | None = None

    @property
    def task_group(self) -> str:
        """Return the ``audio-audiobench-*`` sub-group this task belongs to."""
        return f"audio-audiobench-{self.family}"


def _t(
    upstream_name: str,
    hf_repo: str,
    metric: str,
    family: str,
    *,
    upstream_metric: str | None = None,
    data_dir: str | None = None,
    name: str | None = None,
) -> AudioBenchTaskSpec:
    """Helper — build an :class:`AudioBenchTaskSpec` with sensible defaults.

    By default the canonical name is ``audiobench_<upstream_name>``.  Pass
    ``name`` to override (used when upstream names collide across
    data_dir variants of the same HF repo, e.g. gigaspeech2).
    """
    return AudioBenchTaskSpec(
        name=name if name is not None else _TASK_NAME_PREFIX + upstream_name,
        upstream_name=upstream_name,
        hf_repo=hf_repo,
        metric=metric,
        upstream_metric=upstream_metric or metric,
        family=family,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# Bucket B — 20 genuinely new tasks (not in our lmms-eval task groups)
# ---------------------------------------------------------------------------

_BUCKET_B_ASR = [
    # Mandarin ASR (not in lmms-eval).
    _t("aishell_asr_zh_test", "AudioLLMs/aishell_1_zh_test", "wer", "asr"),
    # Long-form English ASR from financial calls.
    _t("earnings21_test", "AudioLLMs/earnings21_test", "wer", "asr"),
    _t("earnings22_test", "AudioLLMs/earnings22_test", "wer", "asr"),
    # Long-form TED talks (distinct from our tedlium_dev_test).
    _t("tedlium3_long_form_test", "AudioLLMs/tedlium3_long_form_test", "wer", "asr"),
    # GigaSpeech2 — multilingual SE-Asian ASR.  All 3 share one HF repo and
    # are disambiguated by ``data_dir``.  Upstream --dataset name is the same,
    # so we override ``name`` with a language suffix to keep canonical names
    # unique in our CSV.
    _t(
        "gigaspeech2",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        data_dir="th-test",
        name="audiobench_gigaspeech2_thai",
    ),
    _t(
        "gigaspeech2",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        data_dir="id-test",
        name="audiobench_gigaspeech2_indo",
    ),
    _t(
        "gigaspeech2",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        data_dir="vi-test",
        name="audiobench_gigaspeech2_viet",
    ),
    # SEAME code-switch (English ↔ Mandarin).
    _t("seame_dev_man", "AudioLLMs/seame_dev_man", "wer", "asr"),
    _t("seame_dev_sge", "AudioLLMs/seame_dev_sge", "wer", "asr"),
]

_BUCKET_B_ST = [
    # CoVoST2 language pairs not in lmms-eval (only en-zh is there).
    _t("covost2_en_id_test", "AudioLLMs/covost2_en_id_test", "bleu", "st"),
    _t("covost2_en_ta_test", "AudioLLMs/covost2_en_ta_test", "bleu", "st"),
    _t("covost2_id_en_test", "AudioLLMs/covost2_id_en_test", "bleu", "st"),
    _t("covost2_zh_en_test", "AudioLLMs/covost2_zh_en_test", "bleu", "st"),
    _t("covost2_ta_en_test", "AudioLLMs/covost2_ta_en_test", "bleu", "st"),
]

_BUCKET_B_REASONING = [
    # Spoken-MQA reasoning splits (GSM-8K-like, acc scoring).  All 4 share
    # one HF repo; the split is an upstream config — passed as ``data_dir``
    # so the YAML/HF snapshot_download dedups across splits while AudioBench
    # still knows which split to read.
    _t(
        "spoken-mqa",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        data_dir="short_digit",
        name="audiobench_spoken_mqa_short_digit",
    ),
    _t(
        "spoken-mqa",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        data_dir="long_digit",
        name="audiobench_spoken_mqa_long_digit",
    ),
    _t(
        "spoken-mqa",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        data_dir="single_step_reasoning",
        name="audiobench_spoken_mqa_single_step_reasoning",
    ),
    _t(
        "spoken-mqa",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        data_dir="multi_step_reasoning",
        name="audiobench_spoken_mqa_multi_step_reasoning",
    ),
    # MMAU mini — deterministic string-match scoring (judge-free path).
    _t(
        "mmau_mini",
        "AudioLLMs/MMAU-mini",
        "string_match",
        "reasoning",
        upstream_metric="string_match",
    ),
    # AudioCaps — METEOR is the judge-free scorer (judges also available).
    _t(
        "audiocaps_test",
        "AudioLLMs/audiocaps_test",
        "meteor",
        "reasoning",
        upstream_metric="meteor",
    ),
]

# ---------------------------------------------------------------------------
# Bucket A — 7 dual-registered duplicates of benchmarks already in lmms-eval.
# These are for paper-comparability with AudioBench; the lmms-eval versions
# stay in place and produce independent numbers under their own task names.
# The HF repos are distinct (AudioLLMs/* vs lmms-lab/*) so there is no risk
# of snapshot_download collision.
# ---------------------------------------------------------------------------

_BUCKET_A_DUAL = [
    # LibriSpeech (English ASR).
    _t("librispeech_test_clean", "AudioLLMs/librispeech_test_clean", "wer", "asr"),
    _t("librispeech_test_other", "AudioLLMs/librispeech_test_other", "wer", "asr"),
    # Common Voice 15 English ASR.
    _t("common_voice_15_en_test", "AudioLLMs/common_voice_15_en_test", "wer", "asr"),
    # GigaSpeech v1 English ASR.
    _t("gigaspeech_test", "AudioLLMs/gigaspeech_test", "wer", "asr"),
    # People's Speech English ASR (note upstream repo name has the "s").
    _t("peoples_speech_test", "AudioLLMs/peoples_speech_test", "wer", "asr"),
    # TED-LIUM 3 standard test (distinct from tedlium3_long_form_test above).
    _t("tedlium3_test", "AudioLLMs/tedlium3_test", "wer", "asr"),
    # CoVoST2 en→zh (ST).
    _t("covost2_en_zh_test", "AudioLLMs/covost2_en_zh_test", "bleu", "st"),
]


# ---------------------------------------------------------------------------
# Public registry — flat list of all Phase-1 task specs.
# Order is stable (ASR / ST / reasoning) for deterministic YAML ordering
# and for readable test-failure diffs.
# ---------------------------------------------------------------------------

AUDIOBENCH_TASKS: list[AudioBenchTaskSpec] = [
    *_BUCKET_B_ASR,
    *_BUCKET_B_ST,
    *_BUCKET_B_REASONING,
    *_BUCKET_A_DUAL,
]


# Fail-fast consistency checks — runs at import time so a typo in the
# registry breaks the test suite rather than manifesting as a silent job
# routing error later.
def _validate() -> None:
    seen_names: set[str] = set()
    for t in AUDIOBENCH_TASKS:
        if t.name in seen_names:
            raise RuntimeError(f"Duplicate AudioBench task name {t.name!r} in registry")
        seen_names.add(t.name)
        if not t.name.startswith(_TASK_NAME_PREFIX):
            raise RuntimeError(
                f"AudioBench task {t.name!r} missing required prefix "
                f"{_TASK_NAME_PREFIX!r}"
            )
        if t.family not in {"asr", "st", "reasoning"}:
            raise RuntimeError(
                f"AudioBench task {t.name!r} has unknown family {t.family!r}"
            )


_validate()


def get_task_spec(name: str) -> AudioBenchTaskSpec:
    """Look up an :class:`AudioBenchTaskSpec` by canonical task name.

    Raises
    ------
    KeyError
        If *name* does not correspond to any registered AudioBench task.
    """
    for t in AUDIOBENCH_TASKS:
        if t.name == name:
            return t
    known = sorted(t.name for t in AUDIOBENCH_TASKS)
    raise KeyError(f"Unknown AudioBench task {name!r}.  Known tasks: {', '.join(known)}")
