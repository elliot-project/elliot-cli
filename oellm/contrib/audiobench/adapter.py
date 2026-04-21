"""AudioBench model adapter.

Maps a HuggingFace model path (or local filesystem path) to the string key
that AudioBench's ``src/main_evaluate.py --model`` argument expects.  The
upstream dispatch table lives in ``AudioBench/src/model.py`` and is
hand-wired — one entry per model family.

The adapter returns one of:

- ``"qwen2_audio"`` — Qwen2-Audio / Qwen-Audio checkpoints.
- ``"salmonn"``     — SALMONN family (Tsinghua).
- ``"ltu"``         — Listen-Think-Understand.
- ``"whisper"``     — Whisper (OpenAI).
- ``"audioflamingo"`` — Audio-Flamingo (NVIDIA).
- ``"meralion"``    — MERaLiON (Singapore-NLP).
- ``"generic"``     — fallback.  AudioBench treats this as the default HF
                      pipeline dispatch, which works for many generic audio
                      LLMs but may need tuning per model.

The detected value is passed to :mod:`oellm.contrib.dispatch` as the
``model_flags`` portion of the ``eval_suite`` column
(``audiobench:<model_flags>``), exactly like the regiondial_bench pattern.
"""

from __future__ import annotations

from oellm.core.base_model_adapter import BaseModelAdapter

# (model-family key, substrings to match in lowered model path)
# Order matters — first match wins.  More-specific patterns must appear
# before their super-strings (e.g. "qwen2-audio" before "qwen").
_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("qwen2_audio", ("qwen2-audio", "qwen2_audio", "qwen-audio", "qwen_audio")),
    ("salmonn", ("salmonn",)),
    # LTU checkpoints often have paths like "ltu-as/", "ltu-7b", or
    # "MIT/ltu".  Prefix with "/" / "-" where possible to avoid false
    # matches (e.g. "altus").
    ("ltu", ("ltu-", "/ltu", "_ltu", "ltu_as")),
    ("whisper", ("whisper-", "/whisper", "openai/whisper")),
    ("audioflamingo", ("audio-flamingo", "audioflamingo", "audio_flamingo")),
    ("meralion", ("meralion",)),
]


class AudioBenchModelAdapter(BaseModelAdapter):
    """Adapter that resolves ``--model`` flag for AudioBench subprocess."""

    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_lm_eval_args(self) -> str:
        # Not used — AudioBench doesn't route through lm-eval.  Provided
        # only to satisfy the BaseModelAdapter contract.
        return f"pretrained={self._path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        # Not used — see note on to_lm_eval_args().
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        """Return the AudioBench ``--model`` key for this model path."""
        lowered = self._path.lower()
        for key, needles in _PATTERNS:
            if any(n in lowered for n in needles):
                return key
        return "generic"


def detect_audiobench_model_type(model_path: str) -> str:
    """Module-level convenience — matches :func:`oellm.constants.detect_lmms_model_type`.

    Returns the same value as
    ``AudioBenchModelAdapter(model_path).to_contrib_flags()`` but never
    returns ``None`` (falls back to ``"generic"``).
    """
    return AudioBenchModelAdapter(model_path).to_contrib_flags() or "generic"
