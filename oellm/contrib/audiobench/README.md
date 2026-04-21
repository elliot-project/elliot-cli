# AudioBench

AudioBench (AudioLLMs/AudioBench, [arXiv 2406.16020](https://arxiv.org/abs/2406.16020))
is a broad audio-understanding benchmark covering ASR, speech translation,
spoken reasoning, audio scene QA, and paralinguistics. This contrib plugin
wraps AudioBench as a callable `audiobench` suite inside elliot-cli so WP4
can produce numbers directly comparable with the AudioBench paper and
leaderboard, without the scoring-normalisation drift that would come from
running the same datasets through lmms-eval.

## Scope — Phase 1 (this release)

**27 judge-free tasks** across ASR (WER), speech translation (BLEU), spoken
reasoning (accuracy / string_match), and AudioCaps (METEOR). Of these:

- **20 tasks are genuinely new** to the platform — not in any of our
  existing lmms-eval `audio-*` groups. Examples: `earnings21_test`,
  `earnings22_test`, GigaSpeech2 (Thai / Indonesian / Vietnamese),
  SEAME code-switch, Spoken-MQA reasoning splits, MMAU mini.
- **7 tasks are dual-registered** duplicates of benchmarks we already run
  through lmms-eval (LibriSpeech test-clean/other, Common Voice 15 EN,
  GigaSpeech, People's Speech, TED-LIUM 3, CoVoST2 en→zh). These use
  AudioBench's own scorer and normaliser so WP4 can report numbers
  aligned with the AudioBench paper.

Every AudioBench task is namespaced with an `audiobench_` prefix so the CSV
`task_path` column unambiguously identifies which scorer produced a number
(e.g. `audiobench_librispeech_test_clean` is AudioBench-scored;
`librispeech_test_clean` remains the lmms-eval version).

**Phase 2** (not in this release) will add ~19 judge-dependent tasks
(SLUE-SQA5, Spoken-SQuAD, AudioCaps-QA, IEMOCAP / MELD / VoxCeleb probes,
AudioLLM-InstructionFollowing) once a vLLM judge server is provisioned on
Leonardo.

## Prerequisites

### 1. Clone AudioBench on the cluster

AudioBench is **not** pip-installable — upstream is a script harness with
bare imports (`from dataset import ...` inside `src/main_evaluate.py`) and
no `pyproject.toml` / `setup.py`. The plugin invokes it as a subprocess
from an on-cluster clone.

```bash
git clone https://github.com/AudioLLMs/AudioBench /path/to/AudioBench
```

We track the **latest `main`** — no pinned SHA — so updates are a simple
`git pull` under `$AUDIOBENCH_DIR`. If a breaking upstream change lands,
file an issue and we'll introduce a pin.

### 2. Install AudioBench's own runtime dependencies

Still inside the clone:

```bash
cd /path/to/AudioBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

AudioBench's deps (unpinned upstream): `transformers`, `vllm`, `datasets`,
`torchaudio`, `peft`, `autoawq`, `huggingface-hub`, `librosa`, `soundfile`,
`fire`, `evaluate`, `jiwer`, `more_itertools`. Use a **separate venv**
from the elliot-cli venv — AudioBench typically pulls in a bleeding-edge
`transformers` that will conflict with lmms-eval's pin.

### 3. Configure `clusters.yaml`

Add `AUDIOBENCH_DIR` to your cluster block in
`oellm/resources/clusters.yaml`:

```yaml
leonardo:
  ...
  AUDIOBENCH_DIR: "/leonardo/home/userexternal/<user>/AudioBench"
```

The plugin fails fast at dispatch time (via
`oellm.contrib.dispatch`'s `CLUSTER_ENV_VARS` check) if the variable is
missing, so you'll get a clean error message instead of a crash deep
inside the subprocess.

### 4. Install the elliot-cli `audiobench` extra

On the submission / login node where you run `oellm schedule-evals`:

```bash
uv pip install -e ".[audiobench]"
```

This installs our Python-side scorer deps (`jiwer`, `sacrebleu`,
`pythainlp`, `evaluate`) used for result post-processing — **not**
AudioBench itself.

### 5. Dataset pre-download

No manual steps required. `schedule-evals` auto-downloads every
`AudioLLMs/*` HF repo referenced by the requested task group on the
login node via `huggingface_hub.snapshot_download(max_workers=2)` so the
compute nodes do not need internet access. The rate-limit-friendly
`max_workers=2` is shared infrastructure — see `oellm/utils.py`.

## Running

### Available task groups

| Task group                       | Leaves | What it covers                                                  |
|----------------------------------|--------|-----------------------------------------------------------------|
| `audio-audiobench`               | 27     | Full Phase-1 suite (everything below).                          |
| `audio-audiobench-asr`           | 15     | WER tasks — 9 new + 6 dual-registered with lmms-eval.           |
| `audio-audiobench-st`            | 6      | BLEU speech-translation — 5 new + 1 dual (en→zh).               |
| `audio-audiobench-reasoning`     | 6      | Spoken-MQA × 4, MMAU mini, AudioCaps METEOR.                    |

### Example

```bash
# Full AudioBench Phase-1 suite on a Qwen2-Audio model:
oellm schedule-evals \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench \
    --venv-path ~/elliot-venv

# ASR only:
oellm schedule-evals \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench-asr \
    --venv-path ~/elliot-venv

# Smoke test with --limit:
oellm schedule-evals \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench-asr \
    --limit 100 \
    --venv-path ~/elliot-venv
```

`--limit N` is forwarded to AudioBench's `--number_of_samples N`. When
unset, the full test split is evaluated.

### Collecting results

```bash
oellm collect-results \
    --eval-output-dir /path/to/evals \
    --output-csv audiobench_results.csv
```

The primary metric per task is what's registered in `task_metrics`
(`wer` / `bleu` / `accuracy` / `string_match` / `meteor`). Dual-registered
tasks land in the CSV **alongside** their lmms-eval counterparts, with
different `task_path` values (`audiobench_librispeech_test_clean` vs
`librispeech_test_clean`) and different `eval_suite` values (`audiobench`
vs `lmms_eval`) — no silent averaging.

## Supported model adapters

| Model path pattern                  | AudioBench `--model` key |
|-------------------------------------|--------------------------|
| `*qwen2-audio*` / `*qwen-audio*`    | `qwen2_audio`            |
| `*salmonn*`                         | `salmonn`                |
| `*ltu-*` / `*/ltu*` / `*ltu_as*`    | `ltu`                    |
| `*whisper-*` / `*/whisper*`         | `whisper`                |
| `*audio-flamingo*` / `*audioflamingo*` | `audioflamingo`        |
| `*meralion*`                        | `meralion`               |
| (anything else)                     | `generic` (default HF pipeline) |

To override detection explicitly, pass the key as a suffix in the suite
column: `audiobench:qwen2_audio`. The dispatcher in
`oellm/contrib/dispatch.py` already splits on `:`.

## How results flow end-to-end

1. `schedule-evals` expands `audio-audiobench*` groups → 27 rows in
   `jobs.csv` with `eval_suite=audiobench` (plus an adapter suffix from
   `detect_model_flags`).
2. `_collect_dataset_specs` auto-derives `needs_snapshot_download=True`
   from the group-name prefix (`audio-*`) and snapshots every referenced
   `AudioLLMs/*` repo to the shared HF cache.
3. `template.sbatch`'s `*)` catch-all invokes
   `python -m oellm.contrib.dispatch --suite audiobench:<adapter> …`.
4. `oellm.contrib.audiobench.suite.run()` subprocesses
   `python src/main_evaluate.py …` inside `$AUDIOBENCH_DIR`, captures
   the result JSON AudioBench writes under its `--log_dir`, extracts the
   metric value, and writes a lmms-eval-compatible JSON at
   `$output_path`.
5. `collect-results` reads it via `parse_results()` and the standard
   `_resolve_metric` fallback chain — no special-casing in core code.

## Open questions / Phase-2 prerequisites

- **Judge service hosting:** Phase 2 needs a Llama-3-70B-AWQ judge on an
  OpenAI-compatible endpoint. Plan is a separate long-running vLLM sbatch
  whose URL/model lands in `clusters.yaml` as `AUDIOBENCH_JUDGE_URL` and
  `AUDIOBENCH_JUDGE_MODEL`.
- **MERaLiON / IMDA NSC tasks:** ~21 gated AudioBench tasks require
  corpora not on public HF. These will ship in a later phase — or not,
  depending on whether WP4 needs them.
