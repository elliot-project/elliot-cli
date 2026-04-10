"""Shared constants and data structures used across the oellm package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationJob:
    model_path: Path | str
    task_path: str
    n_shot: int
    eval_suite: str


# Mapping of model path patterns to lmms-eval adapter class names.
# Patterns are matched case-insensitively against the model path.
# Order matters: more specific patterns must come before general ones.
LMMS_MODEL_ADAPTERS: list[tuple[list[str], str]] = [
    (["qwen2.5-vl", "qwen2_5_vl", "qwen2.5vl"], "qwen2_5_vl"),
    (["qwen2-vl", "qwen2_vl"], "qwen2_vl"),
    # Video-capable adapters — must precede the generic "llava" pattern
    (["llava-onevision", "llava_onevision"], "llava_onevision"),
    (["llava-vid", "llava_vid", "llava-video"], "llava_vid"),
    (["video-llava", "video_llava"], "video_llava"),
    (["llava"], "llava_hf"),
    (["internvideo"], "internvideo2"),
    (["internvl"], "internvl2"),
    (["idefics"], "idefics3"),
    (["minicpm"], "minicpm_v"),
    (["longva"], "longva"),
    (["videochat2"], "videochat2"),
    (["qwen"], "qwen_vl"),
]

# Fallback metric keys used by collect_results when no task_metrics entry
# is found. Tried in order; first match wins.
METRIC_FALLBACK_KEYS: list[str] = [
    "acc,none",
    "acc",
    "accuracy",
    "f1",
    "exact_match",
]


def detect_lmms_model_type(model_path: str) -> str:
    """Detect the lmms-eval adapter class name from a model path or HF repo name.

    lmms-eval requires --model <adapter_class> (e.g. llava_hf, qwen2_5_vl).
    This is inferred from the model name so users never need to set it manually.

    To add support for a new model family, add an entry to LMMS_MODEL_ADAPTERS
    above or register a BaseModelAdapter via the contrib plugin system.
    """
    name = str(model_path).lower()
    for patterns, adapter in LMMS_MODEL_ADAPTERS:
        if any(p in name for p in patterns):
            return adapter
    raise ValueError(
        f"Cannot auto-detect lmms-eval adapter class from model path '{model_path}'. "
        "Add a pattern to LMMS_MODEL_ADAPTERS in constants.py or register a "
        "BaseModelAdapter via a contrib plugin."
    )
