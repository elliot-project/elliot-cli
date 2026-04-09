# RegionDial-Bench

Multi-round region grounding benchmark on RefCOCOg and RefCOCO+
(Sun et al., ICLR 2026). Evaluates a model's ability to locate and segment
objects described in multi-turn conversations, measuring robustness to error
accumulation across dialogue turns.

**Splits:**
- RefCOCOg Multi-turn — 1,580 images, 4,405 turns
- RefCOCO+ Multi-turn — 715 images, 2,355 turns

**Metrics:** gIoU (primary), cIoU, bbox_AP, pass_rate@0.3/0.5/0.7/0.9,
plus per-round breakdown (R1–R7) for gIoU and bbox_AP.

---

## Prerequisites

### 1. Clone RegionReasoner

The benchmark relies on the inference script
`test/evaluation/evaluation_multi_segmentation.py` and the model wrapper
`test/vision_reasoner/` from the RegionReasoner repository. These are **not
packaged** — the platform calls them directly as a subprocess, so the repo
must be present on the cluster filesystem.

```bash
git clone https://github.com/lmsdss/RegionReasoner \
    /path/to/RegionReasoner
```

### 2. Configure clusters.yaml

Add the following to your cluster entry in `oellm/resources/clusters.yaml`:

```yaml
my-cluster:
  ...
  HF_HOME: "/path/to/large/filesystem/huggingface"   # must have ~30 GB free
  REGION_REASONER_DIR: "/path/to/RegionReasoner"
  GPUS_PER_NODE: 4                                   # controls both SLURM --gres and shard count
```

> **`HF_HOME`** must point to a filesystem with at least **30 GB** of free
> space. On CINECA Leonardo, use the work filesystem
> (`/leonardo_work/<project>/huggingface`), not the home filesystem (50 GB
> quota, fills up quickly).

### 3. Install dependencies in your venv

```bash
# PyTorch — match the CUDA version available on your cluster
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Matching torchvision
pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# flash-attn pre-built wheel (no compilation needed)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# HEIF image support
pip install pi-heif
```

> **flash-attn note:** The pre-built wheel above is for Python 3.12, CUDA 12.x,
> torch 2.5.1. If your configuration differs, find the matching wheel at
> https://github.com/Dao-AILab/flash-attention/releases

### 4. What gets auto-downloaded

When you run `oellm schedule-eval`, the platform automatically pre-downloads
the following on the login node (before SLURM submission, so compute nodes do
not need internet access):

| Asset | HF repo | Size |
|---|---|---|
| TaskRouter-1.5B | `Ricky06662/TaskRouter-1.5B` | ~3 GB |
| SAM2 | `facebook/sam2-hiera-large` | ~1 GB |
| RefCOCOg test JSON | `lmsdss/regionreasoner_test_data` `raw/refcocog_multi_turn.json` | ~26 GB |
| RefCOCOg test images | `lmsdss/regionreasoner_test_data` `raw/refcocog_test_multi_bbox_images/*` | ~200 MB (1 580 images) |
| RefCOCO+ test JSON | `lmsdss/regionreasoner_test_data` `raw/refcocoplus_multi_turn.json` | ~13 GB |
| RefCOCO+ test images | `lmsdss/regionreasoner_test_data` `raw/refcocoplus_test_multi_bbox_images/*` | ~93 MB (715 images) |

All assets are cached under `$HF_HOME/hub`.

---

## Running

Three task groups are available:

| Task group | Splits |
|---|---|
| `regiondial-bench` | Both (RefCOCOg + RefCOCO+) |
| `regiondial-refcocog` | RefCOCOg only (1,580 images, 4,405 turns) |
| `regiondial-refcocoplus` | RefCOCO+ only (715 images, 2,355 turns) |

```bash
# Both splits
oellm schedule-eval \
  --models lmsdss/RegionReasoner-7B \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv

# Single split
oellm schedule-eval \
  --models lmsdss/RegionReasoner-7B \
  --task-groups regiondial-refcocog \
  --venv-path ~/elliot-venv
```

### Collecting results

```bash
oellm collect-results \
  --eval-output-dir /path/to/evals \
  --output-csv results.csv
```

The primary metric in the CSV is **gIoU**. Per-round metrics (e.g.
`gIoU_R1`, `bbox_AP_R3`) are included when the inference script outputs
a `round` field per sample.

---

## Evaluating a different model

The inference script supports multiple model types via the `--model` flag,
which is detected automatically from the model name. To evaluate a different
model, just pass it to `--models`:

```bash
oellm schedule-eval \
  --models Qwen/Qwen2.5-VL-7B-Instruct \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv
```

The model type is resolved as follows:

| Model name pattern | `--model` flag |
|---|---|
| `*regionreasoner*` / `*region_reasoner*` | `vision_reasoner` |
| `*qwen2*` | `qwen2` |
| `*qwen*` | `qwen` |
| anything else | `vision_reasoner` (default) |

To evaluate multiple models in one go:

```bash
oellm schedule-eval \
  --models "lmsdss/RegionReasoner-7B,Qwen/Qwen2.5-VL-7B-Instruct" \
  --task-groups regiondial-bench \
  --venv-path ~/elliot-venv
```

> If your model name does not match any pattern above and requires a specific
> `--model` flag, extend `detect_model_flags()` in `suite.py`.
