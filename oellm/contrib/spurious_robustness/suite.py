"""Spurious-robustness contrib suite — plugin protocol implementation.

Replicates the evaluation of Sarridis et al., "Scaling Vision-Language Models
Fails to Mitigate Bias" (ACM MM 2026), whose reference implementation is at
https://github.com/gsarridis/vlm-spurious-robustness (MIT).

Why this is a contrib suite rather than an lm-eval task: classification here is
image/text *embedding similarity* on an OpenCLIP dual encoder — the image is
assigned to the class whose text embedding is nearest. Neither lm-eval nor
lmms-eval has a similarity-scoring path, and lm-eval's multimodal backends raise
on loglikelihood requests that carry an image, so likelihood-ranked multiple
choice is not an alternative. The benchmark brings its own inference and its own
metric, which is exactly what this plugin interface is for.

Consequence worth stating plainly: this suite evaluates CLIP-family dual
encoders, not the generative VLMs the rest of the platform runs. Scores are
comparable with the paper, and are not comparable with a generatively prompted
model's answers on the same images.

Cluster setup
-------------
``URBANCARS_DATA_DIR``
    Absolute path to the UrbanCars ``test`` tree (the eight
    ``obj-*_bg-*_co_occur_obj-*`` directories). Required only by
    ``spurious_urbancars``; UrbanCars is composited locally rather than
    published, so there is nothing to fetch from the Hub.

``IMAGENET_VAL_DIR``
    Optional. An ILSVRC-2012 validation ImageFolder tree to read instead of the
    gated Hub copy.

Both are deliberately absent from ``CLUSTER_ENV_VARS``: the dispatcher treats
that list as required for every task in the suite, so listing them would make
the CelebA and ImageNet rows fail on clusters that have no UrbanCars tree.
``run()`` raises with the same guidance when a task actually needs one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUITE_NAME = "spurious_robustness"

# Empty by design — see the module docstring.
CLUSTER_ENV_VARS: list[str] = []

from oellm.contrib.spurious_robustness.task import (  # noqa: E402
    SpuriousCelebATask,
    SpuriousImageNetTask,
    SpuriousUrbanCarsTask,
)

_TASKS = (SpuriousImageNetTask(), SpuriousCelebATask(), SpuriousUrbanCarsTask())

_group_kwargs = {"suite": SUITE_NAME, "n_shots": [0]}


def _task_entry(task) -> dict:
    entry: dict = {"task": task.engine_task_name}
    if task.dataset_specs:
        entry["dataset"] = task.dataset_specs[0].repo_id
    return entry


TASK_GROUPS: dict = {
    "task_metrics": {t.engine_task_name: t.primary_metric for t in _TASKS},
    "task_groups": {
        "spurious-robustness": {
            **_group_kwargs,
            "description": (
                "Zero-shot robustness to spurious correlations for OpenCLIP "
                "models: ImageNet, CelebA and UrbanCars."
            ),
            "tasks": [_task_entry(t) for t in _TASKS],
        },
        **{
            t.task_group_name: {
                **_group_kwargs,
                "description": t.description,
                "tasks": [_task_entry(t)],
            }
            for t in _TASKS
        },
    },
}

_EXPECTED_GROUPS = {"spurious_celeba": 4, "spurious_urbancars": 8}


def detect_model_flags(model_path: str) -> str | None:
    """Delegate to OpenClipAdapter.to_contrib_flags()."""
    from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter

    return OpenClipAdapter(model_path).to_contrib_flags()


def _load_model(model_path: str, device: str):
    import open_clip

    from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter

    spec = OpenClipAdapter(model_path).to_open_clip_spec()
    logger.info("Loading OpenCLIP model %s on %s", spec, device)
    model, _, preprocess = open_clip.create_model_and_transforms(spec, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(spec)
    return model, preprocess, tokenizer


def _resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_imagenet(model, preprocess, tokenizer, device, limit, env) -> dict:
    """1000-way zero-shot top-1/top-5, the Radford et al. (2021) protocol.

    The classifier averages all 80 OpenAI templates per class — unlike CelebA and
    UrbanCars, which take the maximum over their prompts.
    """
    import torch
    from open_clip import (
        IMAGENET_CLASSNAMES,
        OPENAI_IMAGENET_TEMPLATES,
        build_zero_shot_classifier,
    )

    from oellm.contrib.spurious_robustness.datasets import load_imagenet
    from oellm.contrib.spurious_robustness.zeroshot import encode_images

    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=False,
    )

    top1 = top5 = total = 0
    for images, labels, _ in load_imagenet(env.get("IMAGENET_VAL_DIR") or None, limit):
        features = encode_images(model, preprocess, images, device)
        logits = 100.0 * features.to(device) @ classifier
        targets = torch.tensor(labels, device=device)
        _, ranked = logits.topk(5, dim=-1)
        hits = ranked == targets.view(-1, 1)
        top1 += hits[:, :1].sum().item()
        top5 += hits.sum().item()
        total += len(labels)

    if not total:
        raise RuntimeError("ImageNet evaluation produced no samples")

    return {
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        # One group by construction: no spurious attribute to split on.
        "worst_group_accuracy": top1 / total,
        "n_groups": 1,
        "total_images": total,
    }


def _run_grouped(model, preprocess, tokenizer, device, batches, prompts, class_names):
    """Shared path for the two grouped benchmarks (max-over-prompts scoring)."""
    from oellm.contrib.spurious_robustness.zeroshot import (
        class_scores,
        encode_images,
        encode_texts,
        group_metrics,
        predict,
    )

    prompt_features = [
        encode_texts(model, tokenizer, prompts[c], device) for c in class_names
    ]

    predictions: list[int] = []
    labels: list[int] = []
    groups: list[str] = []
    for images, batch_labels, batch_groups in batches:
        features = encode_images(model, preprocess, images, device)
        predictions.extend(predict(class_scores(features, prompt_features)).tolist())
        labels.extend(batch_labels)
        groups.extend(batch_groups)

    metrics = group_metrics(predictions, labels, groups)
    per_group = metrics.pop("group_accuracies")
    metrics.pop("group_counts")
    for group, acc in sorted(per_group.items()):
        metrics[f"acc_{group}"] = acc
    return metrics


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
    from oellm.contrib.spurious_robustness.datasets import (
        CELEBA_CLASS_NAMES,
        URBANCARS_CLASS_NAMES,
        load_celeba,
        load_urbancars,
    )
    from oellm.contrib.spurious_robustness.prompts import (
        CELEBA_PROMPTS,
        URBANCARS_PROMPTS,
    )

    known = [t.engine_task_name for t in _TASKS]
    if task not in known:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {known}")

    # Resolve everything that can fail from configuration alone before loading
    # the model — a missing data directory should not cost a checkpoint load.
    data_dir = env.get("URBANCARS_DATA_DIR", "")
    if task == "spurious_urbancars" and not data_dir:
        raise RuntimeError(
            "URBANCARS_DATA_DIR must be set for spurious_urbancars. UrbanCars "
            "is composited locally (Stanford Cars + Places + LVIS via "
            "Whac-A-Mole) rather than published, so it cannot be fetched from "
            "the Hub. Point it at the bg-0.5_co_occur_obj-0.5 test tree."
        )

    limit_raw = env.get("LIMIT", "").strip()
    limit = int(limit_raw) if limit_raw else None
    device = _resolve_device()
    model, preprocess, tokenizer = _load_model(model_path, device)

    if task == "spurious_imagenet":
        metrics = _run_imagenet(model, preprocess, tokenizer, device, limit, env)
    elif task == "spurious_celeba":
        metrics = _run_grouped(
            model,
            preprocess,
            tokenizer,
            device,
            load_celeba(limit),
            CELEBA_PROMPTS,
            CELEBA_CLASS_NAMES,
        )
    else:
        metrics = _run_grouped(
            model,
            preprocess,
            tokenizer,
            device,
            load_urbancars(data_dir, limit),
            URBANCARS_PROMPTS,
            URBANCARS_CLASS_NAMES,
        )

    expected = _EXPECTED_GROUPS.get(task)
    observed = metrics.get("n_groups")
    if expected is not None and observed != expected:
        # Worst-group is a minimum, so a missing group can only make the score
        # look better. Never let that pass silently.
        logger.warning(
            "%s covered %s of %d groups — worst-group accuracy is a minimum over "
            "the groups present, so this number is not comparable with a full run.",
            task,
            observed,
            expected,
        )

    result_json = {
        "model_name_or_path": model_path,
        "results": {task: metrics},
        "configs": {task: {"num_fewshot": n_shot}},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info("Results written to %s", output_path)


def parse_results(data: dict) -> tuple[str, str, int, dict[str, float]] | None:
    """Claim *data* if it is this suite's output, else return None."""
    results = data.get("results", {})
    for task_name, task_results in results.items():
        if task_name.startswith("spurious_") and isinstance(task_results, dict):
            if "worst_group_accuracy" not in task_results:
                continue
            model_id = data.get("model_name_or_path") or data.get("model_name", "unknown")
            n_shot = data.get("configs", {}).get(task_name, {}).get("num_fewshot", 0)
            return model_id, task_name, int(n_shot), task_results
    return None
