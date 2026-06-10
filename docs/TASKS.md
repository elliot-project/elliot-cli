# Adding Tasks and Task Groups

## Overview

Tasks are defined in `oellm/resources/task-groups.yaml`. Only tasks in this file are tested and guaranteed to work. The CLI parses this via `task_groups.py` and expands groups into `(task, n_shot, suite)` tuples for scheduling.

Supported evaluation suites:

| Suite value | Engine | Use case |
|---|---|---|
| `lm_eval` (alias `lm-eval-harness`) | [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) | Text benchmarks |
| `lighteval` | [lighteval](https://github.com/huggingface/lighteval) | Translation / multilingual |
| `lmms_eval` | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Image / video / audio benchmarks |
| `evalchemy` | [evalchemy](https://github.com/mlfoundations/evalchemy) (fork) | Free-form reasoning (GPQA, MATH500, LiveCodeBench) — needs its own venv, see [VENV.md](VENV.md) |
| `<contrib name>` | a contrib plugin | Custom benchmarks — see [CONTRIBUTING.md](../oellm/contrib/CONTRIBUTING.md) |

`suite` is set at the group level and can be overridden per task — the
`reasoning` group uses this to mix lm-eval and evalchemy tasks in one group.

## YAML Structure

```yaml
task_groups:
  my-group:
    description: "Short description"
    suite: lm_eval        # or lighteval, lmms_eval
    n_shots: [5]          # default for all tasks in group
    dataset: org/dataset  # default HF dataset for pre-download
    tasks:
      - task: task_name
        n_shots: [0, 5]     # overrides group default
        dataset: org/other  # overrides group default
        subset: subset_name # HF dataset config/subset
```

## Adding a Text Task Group

1. Add your group to `oellm/resources/task-groups.yaml`:

```yaml
task_groups:
  my-benchmark:
    description: "My custom benchmark"
    suite: lm_eval
    n_shots: [0]
    dataset: huggingface/dataset-name
    tasks:
      - task: task_one
        subset: split_a
      - task: task_two
        subset: split_b
```

2. Use it:

```bash
oellm-eval schedule --models "model-name" --task-groups "my-benchmark"
```

## Adding an Image Task Group

Image tasks use `suite: lmms_eval`. Each task must correspond to a task name recognized by lmms-eval.

```yaml
task_groups:
  my-image-benchmark:
    description: "My image benchmark via lmms-eval"
    suite: lmms_eval
    n_shots: [0]
    tasks:
      - task: vqav2_val_all
        dataset: HuggingFaceM4/VQAv2
      - task: mmbench_en_dev
        dataset: HuggingFaceM4/MMBench_00
```

Run with:

```bash
oellm-eval schedule --models "path/to/vlm" --task-groups "my-image-benchmark"
```

The lmms-eval model adapter (e.g. `llava_hf`, `qwen2_vl`) is auto-detected
from the model name. No manual override is needed.

## Field Reference

| Field | Required | Level | Description |
|-------|----------|-------|-------------|
| `description` | Yes | group | Short description of the task group |
| `suite` | Yes (group) | group or task | Evaluation suite; a task-level value overrides the group |
| `n_shots` | Yes | group or task | List of shot counts; must be set at group or task level |
| `dataset` | Recommended | group or task | HuggingFace dataset repo ID — without it, nothing is pre-downloaded for the task |
| `task` | Yes | task | Task name as recognized by the evaluation suite |
| `subset` | No | task | HuggingFace dataset config/subset name |
| `revisions` | No | task | HF dataset revisions/branches to pre-fetch (default `["main"]`; e.g. MVBench keeps videos on a `video` branch) |
| `hf_models` | No | task | Auxiliary HF *model* repos to pre-download (e.g. a task router or SAM checkpoint) |
| `hf_dataset_files` | No | task | Specific files to fetch from a dataset repo: `{repo_id, patterns, revision?}` — use for large repos where only a subset is needed |

## Important: Dataset Pre-Download Behavior

**Provide the `dataset` field** (at group or task level) wherever possible:
compute nodes run with `HF_HUB_OFFLINE=1`, so anything not cached on the
login node before submission fails at eval time. Tasks without a `dataset`
field are simply skipped by the pre-download step.

Two details worth knowing:

1. **The group-name prefix selects the download strategy.** Groups whose name
   starts with `audio-`, `video-`, or `image-` are fetched with
   `snapshot_download` (raw repo files; the compute node builds the dataset at
   runtime — this avoids out-of-memory kills on the login node for large
   media datasets). All other groups go through `load_dataset()`. If you add
   a media group, keep the prefix.
2. **Dataset accessibility is verified by `tests/test_datasets.py`**, which
   needs network access and is therefore excluded from the offline CI run —
   execute it locally when adding datasets:
   `uv run pytest tests/test_datasets.py -k my-group`.

## Custom Benchmarks (contrib plugins)

For benchmarks that have their own inference scripts, custom metrics, or
are not part of lm-eval / lighteval / lmms-eval, use the contrib plugin
system. See [`oellm/contrib/CONTRIBUTING.md`](../oellm/contrib/CONTRIBUTING.md)
for the full guide and `oellm/contrib/regiondial_bench/` as a reference
implementation.
