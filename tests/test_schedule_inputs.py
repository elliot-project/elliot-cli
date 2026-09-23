"""What gets scheduled, and what the generated job script runs, for a given
set of CLI inputs and login-shell environment."""

import csv
import os
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals
from oellm.task_groups import _expand_task_groups


def _schedule(tmp_path, env=None, **kw):
    """Dry-run schedule; return the rendered job script and jobs.csv rows."""
    out = tmp_path / "out"
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(out), **(env or {})}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m", dry_run=True, skip_checks=True, **kw
        )
    run = next(out.iterdir())
    with open(run / "jobs.csv") as f:
        rows = list(csv.DictReader(f))
    return (run / "submit_evals.sbatch").read_text(), rows


def _venv(tmp_path, name):
    venv = tmp_path / name
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("")
    return str(venv)


class TestLoginShellVariables:
    def test_exported_job_variables_do_not_override_flags(self, tmp_path):
        """VENV_PATH, LIMIT and MODEL_DIR are the job script's own variables;
        exported in the login shell they used to be baked into the script."""
        chosen = _venv(tmp_path, "chosen")
        sbatch, _ = _schedule(
            tmp_path,
            env={
                "VENV_PATH": "/stray/venv",
                "LIMIT": "100",
                "MODEL_DIR": "/stray/models",
            },
            tasks="copa",
            n_shot=[0],
            venv_path=chosen,
            limit=5,
        )
        assert "/stray" not in sbatch
        assert f'VENV_PATH="{chosen}"' in sbatch
        assert 'source "$VENV_PATH/bin/activate"' in sbatch
        assert 'export LIMIT="5"' in sbatch
        assert "${LIMIT:+--limit $LIMIT}" in sbatch

    def test_cluster_settings_are_still_filled_in(self, tmp_path):
        sbatch, _ = _schedule(
            tmp_path,
            env={"PARTITION": "boost_usr_prod", "GPUS_PER_NODE": "1", "HF_HOME": "/hf"},
            tasks="copa",
            n_shot=[0],
        )
        assert "#SBATCH --partition=boost_usr_prod" in sbatch
        assert "#SBATCH --gres=gpu:1" in sbatch
        assert "export HF_HOME=/hf" in sbatch


class TestTasksAndGroups:
    def test_tasks_and_groups_are_both_scheduled(self, tmp_path):
        _, rows = _schedule(
            tmp_path, tasks="gsm8k", n_shot=[0], task_groups="open-sci-0.01"
        )
        tasks = {(r["task_path"], r["n_shot"]) for r in rows}
        assert ("gsm8k", "0") in tasks
        assert len(rows) == len(_expand_task_groups(["open-sci-0.01"])) + 1

    def test_a_task_already_in_the_group_is_scheduled_once(self, tmp_path):
        _, rows = _schedule(
            tmp_path, tasks="hellaswag", n_shot=[10], task_groups="open-sci-0.01"
        )
        assert [r["n_shot"] for r in rows if r["task_path"] == "hellaswag"] == ["10"]

    def test_n_shot_with_groups_only_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="n_shot applies to tasks only"):
            _schedule(tmp_path, n_shot=[0], task_groups="open-sci-0.01")


class TestSharedTasks:
    @pytest.mark.parametrize(
        "groups",
        [["open-sci-0.01", "dclm-core-22"], ["dclm-core-22", "open-sci-0.01"]],
    )
    def test_every_groups_shots_are_kept(self, groups):
        """hellaswag is 10-shot in open-sci-0.01 and 0- and 10-shot in
        dclm-core-22; with open-sci-0.01 listed first, the 0-shot run was
        dropped (the reverse order always worked)."""
        shots = {r.n_shot for r in _expand_task_groups(groups) if r.task == "hellaswag"}
        assert shots == {0, 10}

    def test_groups_are_not_modified(self):
        _expand_task_groups(["open-sci-0.01", "dclm-core-22"])
        shots = {
            r.n_shot
            for r in _expand_task_groups(["open-sci-0.01"])
            if r.task == "hellaswag"
        }
        assert shots == {10}


class TestDataStaging:
    def test_tasks_next_to_groups_get_their_data_staged(self, tmp_path, monkeypatch):
        from oellm.task_groups import _collect_dataset_specs

        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("")
        staged = []
        monkeypatch.setattr(
            "oellm.envcheck.check_scheduled_environment", lambda *a, **k: None
        )
        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch("oellm.scheduler._ensure_runtime_environment"),
            patch("oellm.scheduler._process_model_paths", return_value={}),
            patch("oellm.scheduler._probe_engine_versions", return_value={}),
            patch(
                "oellm.scheduler._pre_download_datasets_from_specs",
                side_effect=lambda specs, **_: staged.extend(specs),
            ),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path / "out")}),
        ):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="xcopa",
                n_shot=[0],
                task_groups="open-sci-0.01",
                dry_run=True,
                venv_path=str(venv),
            )
        repos = [spec.repo_id for spec in staged]
        assert "xcopa" in repos
        assert set(repos) >= {
            s.repo_id for s in _collect_dataset_specs(["open-sci-0.01"])
        }
        specs = [(spec.repo_id, spec.subset) for spec in staged]
        assert len(specs) == len(set(specs))  # nothing staged twice


class TestDatasetsCache:
    def test_the_job_uses_the_cache_the_pre_download_filled(self, tmp_path):
        """datasets honours an exported HF_DATASETS_CACHE when pre-downloading,
        so the job must read from there too (sbatch passes the environment)."""
        import subprocess

        sbatch, _ = _schedule(tmp_path, env={"HF_HOME": "/hf"}, tasks="copa", n_shot=[0])
        line = next(
            ln for ln in sbatch.splitlines() if ln.startswith("export HF_DATASETS_CACHE=")
        )
        assert line == 'export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/hf/datasets}"'
        for exported, expected in (
            ("/scratch/ds", "/scratch/ds"),
            (None, "/hf/datasets"),
        ):
            env = {"PATH": os.environ["PATH"]}
            if exported:
                env["HF_DATASETS_CACHE"] = exported
            out = subprocess.run(
                ["bash", "-c", f'{line}; echo "$HF_DATASETS_CACHE"'],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            assert out.stdout.strip() == expected
