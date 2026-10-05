import os
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from oellm.main import schedule_evals

_config = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
ALL_TASK_GROUPS = list(_config["task_groups"].keys())


@pytest.mark.parametrize("n_shot", [None, 0])
@pytest.mark.parametrize("task_groups", ALL_TASK_GROUPS)
def test_schedule_evals(tmp_path, n_shot, task_groups):
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups=task_groups,
            n_shot=n_shot,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )


def test_tasks_are_scheduled_next_to_task_groups(tmp_path):
    """`tasks` given together with `task_groups` are scheduled too, each pair once."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            # the group already schedules crows_pairs_english at 0-shot
            tasks="hellaswag,crows_pairs_english",
            task_groups="crows-pairs",
            n_shot=[0, 5],
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )

    df = pd.read_csv(next(iter(tmp_path.glob("**/jobs.csv"))))
    scheduled = sorted(df[["task_path", "n_shot"]].itertuples(index=False, name=None))
    assert scheduled == [
        ("crows_pairs_english", 0),
        ("crows_pairs_english", 5),
        ("hellaswag", 0),
        ("hellaswag", 5),
    ]


def test_datasets_of_tasks_and_task_groups_are_pre_downloaded(tmp_path):
    """Data for `tasks` is staged too when they are given next to `task_groups`."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._ensure_runtime_environment"),
        patch("oellm.main._process_model_paths"),
        patch("oellm.main._pre_download_datasets_from_specs") as pre_download,
        patch.dict(
            os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "HF_HOME": str(tmp_path)}
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag,crows_pairs_english",
            task_groups="crows-pairs",
            n_shot=0,
            venv_path=str(Path(sys.prefix)),
            download_only=True,
        )

    specs = pre_download.call_args.args[0]
    assert [(spec.repo_id, spec.subset) for spec in specs] == [
        ("jannalu/crows_pairs_multilingual", "english"),
        ("Rowan/hellaswag", None),
    ]


@pytest.mark.parametrize("task_groups", [None, "crows-pairs"])
def test_tasks_without_n_shot_fail_before_any_download(tmp_path, task_groups):
    """`tasks` need `n_shot`, and the error comes before the runtime check."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._ensure_runtime_environment") as runtime_check,
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        with pytest.raises(ValueError, match="`n_shot` is required"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                task_groups=task_groups,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )
    runtime_check.assert_not_called()


def test_schedule_evals_slurm_template_var_overrides(tmp_path):
    """Verify --slurm_template_var JSON overrides appear in the generated sbatch."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {
                "EVAL_OUTPUT_DIR": str(tmp_path),
                "PARTITION": "default_partition",
                "ACCOUNT": "test_account",
            },
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            slurm_template_var='{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:15:00","GPUS_PER_NODE":2}',
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --partition=dev-g" in sbatch_content
    assert "#SBATCH --account=myproject" in sbatch_content
    assert "#SBATCH --time=02:15:00" in sbatch_content
    assert "#SBATCH --gres=gpu:2" in sbatch_content


def test_schedule_evals_nodelist(tmp_path):
    """Verify --nodelist adds an #SBATCH --nodelist directive to the sbatch."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {**env, "EVAL_OUTPUT_DIR": str(tmp_path)}, clear=True),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            nodelist="tdll-3gpu4",
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --nodelist=tdll-3gpu4" in sbatch_content


def test_schedule_evals_no_nodelist(tmp_path):
    """Without --nodelist the directive is stripped from the sbatch."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {**env, "EVAL_OUTPUT_DIR": str(tmp_path)}, clear=True),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "--nodelist" not in sbatch_content


def test_schedule_evals_slurm_template_var_invalid_json(tmp_path):
    """Verify invalid slurm_template_var raises ValueError."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        with pytest.raises(ValueError, match="valid JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var="not valid json",
            )
        with pytest.raises(ValueError, match="must be a JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var='["partition", "dev-g"]',
            )
