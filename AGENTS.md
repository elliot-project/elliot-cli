# AGENTS.md

Guidance for coding agents and new contributors working on this repository.
Humans: start with [README.md](README.md); this file collects the non-obvious
rules that are easy to violate.

## Setup and commands

- Install: `uv sync --extra dev` (Python >= 3.12).
- Tests (the offline baseline — run before claiming anything works):

  ```bash
  uv run pytest tests/ -q --ignore=tests/integration --deselect tests/test_datasets.py
  ```

  `tests/test_datasets.py` and `tests/integration/` need network and HF access;
  everything else runs offline.
- Lint and format: `uvx ruff check oellm/ tests/ scripts/` and
  `uvx ruff format --check oellm/ tests/ scripts/`.
- The evaluation engines (lm-eval, lighteval, lmms-eval, evalchemy) do NOT live
  in this project's venv — they live in separate engine venvs described in
  [docs/VENV.md](docs/VENV.md). lmms-eval must be installed editable from git
  (its wheel build is broken); it is deliberately not a pip extra.

## Invariants — do not change casually

- `jobs.csv` schema is frozen: `model_path, task_path, n_shot, eval_suite`.
- The plugin protocol is frozen at `oellm.core.CORE_API_VERSION = "1.0"`;
  `tests/test_plugin_protocol.py` is the executable spec — if a protocol member
  loses its consumer, that file must fail.
- `oellm/resources/template.sbatch` is rendered with Python `str.format` and
  then `Template.safe_substitute`. Bash `${VAR}` must be written `${{VAR}}`,
  and any literal `{` or `}` — **including inside comments** — crashes
  scheduling with `KeyError`. Never put JSON examples in template comments.
- Every task gets an explicit `task_metrics` entry (the headline metric is
  pinned in advance; the scheduler warns about unmapped tasks). Changing a
  pinned metric shifts collected numbers — call out the re-baselining need in
  the PR description.
- YAML keeps the last duplicate key silently. After merges touching
  `task-groups.yaml`, check for duplicate keys — this has bitten before.
- Valid `eval_suite` values: `lm_eval` (default), `lighteval`, `lmms_eval`,
  `evalchemy`, or a contrib `SUITE_NAME`.

## Configuration hygiene

- `oellm/resources/clusters.yaml` is shared configuration. Never commit
  personal accounts, home paths, or cache locations. Personal overrides go
  through environment variables — precedence is user env > cluster values >
  shared values, including inside templated values like
  `EVAL_OUTPUT_DIR: "{EVAL_BASE_DIR}/{USER}"`.
- `uv.lock` is deliberately untracked: it pins only the orchestrator's own
  environment, not the engine environments that produce scores.

## Adding a benchmark — three paths, lightest first

1. Already in lm-eval / lighteval / lmms-eval: a YAML entry in
   `task-groups.yaml` (group + `task_metrics` pin).
2. Not in any engine but fits lm-eval's task model: a custom task under
   `oellm/resources/custom_lm_eval_tasks/` — references: `tabfact/`,
   `timeseriesexam/`. No plugin, no core changes.
3. Own inference script, custom metrics, or sharding: a contrib plugin under
   `oellm/contrib/`.

Details in [oellm/contrib/CONTRIBUTING.md](oellm/contrib/CONTRIBUTING.md).
For paths 2 and 3, add a conformance test class to
`tests/test_plugin_protocol.py` (group expansion, metric mapping, prompt
construction on a synthetic doc).

## Verifying changes

- Render level, no cluster needed: schedule with `--dry-run`, then read the
  generated `.sbatch` (and `bash -n` it), `jobs.csv`, and `provenance.json`.
- On a cluster, smoke-test with a tiny model:

  ```bash
  ROW_TIMEOUT=30m oellm-eval schedule --models EleutherAI/pythia-70m \
      --tasks hellaswag --n-shot 0 --limit 10 --venv-path /path/to/.venv
  oellm-eval collect <run_dir>
  ```

  Then read `eval_results.md`: the metric column must match the pinned
  `task_metrics` entry, and `provenance.json` must record engine versions.
- Results produced with `--limit` or quantization are test artifacts: they are
  badged and excluded from aggregates downstream. Never compare them against
  full-precision, full-dataset runs.

## Deployment

Code reaches clusters via git (clone or pull on the login node). Cluster
containers are built and pushed through the GitHub Actions workflow — see
"Deploying Containers" in the README.
