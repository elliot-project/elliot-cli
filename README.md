# ELLIOT Evaluation Platform

A multimodal evaluation framework for scheduling LLM and VLM evaluations across HPC clusters. Built as an orchestration layer over [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/huggingface/lighteval), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), with a plugin system for contributing custom benchmarks.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm schedule-eval`
- **Collect results** and check for missing evaluations: `oellm collect-results`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA)
- **Image evaluation** via lmms-eval (VQAv2, MMBench, MMMU, ChartQA, DocVQA, TextVQA, OCRBench, MathVista)
- **Plugin system** for contributing custom benchmarks without touching core code
- **Automatic container builds** via GitHub Actions

## Quick Start

**Prerequisites:**
- Install [uv](https://docs.astral.sh/uv/#installation)
- Set `HF_HOME` to your HuggingFace cache directory (e.g. `export HF_HOME="/path/to/hf_home"`)

```bash
# Install
uv tool install -p 3.12 git+https://github.com/elliot-project/elliot-cli.git

# Run evaluations using a task group
oellm schedule-eval \
    --models "EleutherAI/pythia-160m" \
    --task_groups "open-sci-0.01"

# Image evaluation (requires venv with lmms-eval)
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task_groups "image-vqa" \
    --venv_path ~/elliot-venv
```

This will automatically detect your cluster, download models and datasets, and submit a SLURM job array with cluster-specific resources.

For custom environments instead of containers, pass `--venv_path` (see [docs/VENV.md](docs/VENV.md)).

## Task Groups

Task groups are pre-defined evaluation suites in [`task-groups.yaml`](oellm/resources/task-groups.yaml). Each group specifies tasks, their n-shot settings, and HuggingFace dataset mappings.

### Text & Multilingual

| Group | Description | Engine |
|---|---|---|
| `open-sci-0.01` | COPA, MMLU, HellaSwag, ARC, etc. | lm-eval |
| `belebele-eu-5-shot` | Belebele in 23 European languages | lm-eval |
| `flores-200-eu-to-eng` | EU to English translation | lighteval |
| `flores-200-eng-to-eu` | English to EU translation | lighteval |
| `global-mmlu-eu` | Global MMLU in EU languages | lm-eval |
| `mgsm-eu` | Multilingual GSM8K | lm-eval |
| `generic-multilingual` | XWinograd, XCOPA, XStoryCloze | lm-eval |
| `include` | INCLUDE benchmarks (44 languages) | lm-eval |

Super groups: `oellm-multilingual` (all multilingual benchmarks combined)

### Image

| Group | Benchmark | Engine |
|---|---|---|
| `image-vqa` | All 8 benchmarks combined | lmms-eval |
| `image-vqav2` | VQAv2 | lmms-eval |
| `image-mmbench` | MMBench | lmms-eval |
| `image-mmmu` | MMMU | lmms-eval |
| `image-chartqa` | ChartQA | lmms-eval |
| `image-docvqa` | DocVQA | lmms-eval |
| `image-textvqa` | TextVQA | lmms-eval |
| `image-ocrbench` | OCRBench | lmms-eval |
| `image-mathvista` | MathVista | lmms-eval |

The lmms-eval adapter class (`llava_hf`, `qwen2_5_vl`, etc.) is auto-detected from the model name.

### Custom Benchmarks (contrib)

Community-contributed benchmarks that run outside the standard evaluation engines. See the [contrib registry](oellm/contrib/README.md) for the full list.

```bash
# Run all 8 image benchmarks
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task_groups "image-vqa" \
    --venv_path ~/elliot-venv

# Mix image and text benchmarks in one submission
oellm schedule-eval \
    --models "llava-hf/llava-1.5-7b-hf" \
    --task_groups "image-mmbench,open-sci-0.01" \
    --venv_path ~/elliot-venv

# Use multiple task groups or a super group
oellm schedule-eval --models "model-name" --task_groups "belebele-eu-5-shot,global-mmlu-eu"
oellm schedule-eval --models "model-name" --task_groups "oellm-multilingual"
```

## Collecting Results

```bash
# Basic collection
oellm collect-results /path/to/eval-output-dir

# Check for missing evaluations and create a CSV for re-running them
oellm collect-results /path/to/eval-output-dir --check --output_csv results.csv

# Re-schedule failed jobs
oellm schedule-eval --eval_csv_path results_missing.csv
```

## Installation

```bash
uv tool install -p 3.12 git+https://github.com/elliot-project/elliot-cli.git
```

Update to latest:
```bash
uv tool upgrade oellm
```

For cluster-specific setup, see the [documentation](#documentation) section.

## Development

```bash
git clone https://github.com/elliot-project/elliot-cli.git
cd elliot-cli
uv sync --extra dev

# Run all unit tests
uv run pytest tests/ -v

# Download-only mode for testing
uv run oellm schedule-eval --models "EleutherAI/pythia-160m" --task_groups "open-sci-0.01" --download_only
```

## Documentation

### Cluster Setup

| Cluster | Guide |
|---|---|
| Leonardo (CINECA) | [docs/LEONARDO.md](docs/LEONARDO.md) |
| LUMI, JURECA | Coming soon |

### Environment & Infrastructure

| Doc | Description |
|---|---|
| [Using a Virtual Environment](docs/VENV.md) | Setting up a custom venv with lm-eval, lmms-eval, and lighteval |
| [Container Workflow](docs/CONTAINERS.md) | How Apptainer containers are built, deployed, and used |

### Extending the Platform

| Doc | Description |
|---|---|
| [Adding Tasks & Task Groups](docs/TASKS.md) | YAML structure for defining new evaluation suites |
| [Contributing Custom Benchmarks](oellm/contrib/CONTRIBUTING.md) | Step-by-step guide for adding a contrib plugin |
| [Contrib Registry](oellm/contrib/README.md) | List of community-contributed benchmarks |

## Contributing Custom Benchmarks

ELLIOT supports two paths for adding benchmarks:

1. **Benchmark already in lm-eval / lighteval / lmms-eval** -- add a YAML entry to [`task-groups.yaml`](oellm/resources/task-groups.yaml)
2. **Fully custom benchmark** -- drop a contrib plugin into [`oellm/contrib/`](oellm/contrib/)

See the [Contributing Guide](oellm/contrib/CONTRIBUTING.md) for step-by-step instructions.

## Deploying Containers

Containers are deployed manually since [PR #46](https://github.com/elliot-project/elliot-cli/pull/46). To build and deploy, select "Run workflow" in [Actions](https://github.com/elliot-project/elliot-cli/actions/workflows/build-and-push-apptainer.yml).
