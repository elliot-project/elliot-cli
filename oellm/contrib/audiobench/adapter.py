"""AudioBench model adapter.

Maps a HuggingFace model path to the string key that AudioBench's
``src/main_evaluate.py --model`` argument expects.  The detected value is
passed to :mod:`oellm.contrib.dispatch` as the ``model_flags`` portion of
the ``eval_suite`` column (``audiobench:<model_flags>``).
"""

from __future__ import annotations

from oellm.core.base_model_adapter import BaseModelAdapter

# (model-family key, substrings to match in lowered model path).  Order
# matters — first match wins, so more-specific patterns come first.
_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("qwen2_audio", ("qwen2-audio", "qwen2_audio", "qwen-audio", "qwen_audio")),
    ("salmonn", ("salmonn",)),
    ("ltu", ("ltu-", "/ltu", "_ltu", "ltu_as")),
    ("whisper", ("whisper-", "/whisper", "openai/whisper")),
    ("audioflamingo", ("audio-flamingo", "audioflamingo", "audio_flamingo")),
    ("meralion", ("meralion",)),
]


class AudioBenchModelAdapter(BaseModelAdapter):
    """Adapter resolving the ``--model`` flag for the AudioBench subprocess."""

    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_lm_eval_args(self) -> str:
        # Unused — AudioBench doesn't route through lm-eval.  Required by
        # BaseModelAdapter.
        return f"pretrained={self._path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        # Unused — see to_lm_eval_args().
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        lowered = self._path.lower()
        for key, needles in _PATTERNS:
            if any(n in lowered for n in needles):
                return key
        return "generic"


def detect_audiobench_model_type(model_path: str) -> str:
    """Like ``to_contrib_flags`` but always returns a string (default ``generic``)."""
    return AudioBenchModelAdapter(model_path).to_contrib_flags() or "generic"
