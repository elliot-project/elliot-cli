"""Generation for DocVQA 2026: one prompt per question, all pages attached."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 512


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_limit(env: dict[str, str]) -> int | None:
    raw = str(env.get("LIMIT", "")).strip()
    return int(raw) if raw else None


def load_model(model_path: str, device: str):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    logger.info("Loading vision-language model %s on %s", model_path, device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def build_messages(question: str, n_images: int, prompt: str) -> list[dict]:
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": f"{prompt}\n\nQUESTION: {question}"})
    return [{"role": "user", "content": content}]


def generate_answer(model, processor, sample, prompt: str, device: str) -> str:
    """Raw model output for one question — parsing belongs to the scorer."""
    import torch

    messages = build_messages(sample.question, len(sample.images), prompt)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=sample.images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(generated[0][prompt_len:], skip_special_tokens=True)


def write_results(
    output_path: Path, model_path: str, task: str, n_shot: int, metrics: dict
) -> None:
    result_json = {
        "model_name_or_path": model_path,
        "results": {task: metrics},
        "configs": {task: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info("Results written to %s", output_path)


def parse_suite_results(
    data: dict, task_prefix: str
) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it carries this suite's task and metric shape."""
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if not task_name.startswith(task_prefix) or not isinstance(task_results, dict):
            continue
        if "macro_accuracy" not in task_results:
            continue
        model_id = data.get("model_name_or_path") or data.get("model_name", "unknown")
        n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
        return (model_id, task_name, int(n_shot), dict(task_results))
    return None
