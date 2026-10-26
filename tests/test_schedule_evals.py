import os
import shlex
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oellm.main import schedule_evals

_config = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
ALL_TASK_GROUPS = list(_config["task_groups"].keys())


@pytest.fixture
def schedule_env(tmp_path):
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch.dict(
            os.environ,
            {"EVAL_OUTPUT_DIR": str(tmp_path), "HF_HOME": str(tmp_path / "hf")},
        ),
    ):
        yield


@pytest.mark.parametrize("task_groups", ALL_TASK_GROUPS)
def test_schedule_evals(tmp_path, task_groups):
    # n_shot with a group is refused (test_schedule_inputs.py).
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.runner.detect_lmms_model_type", return_value="llava"),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        # ``allow_missing_judge`` so this parametrized smoke test can exercise
        # judge-required task groups (audio-alpaca-audio, video-activitynet-qa,
        # …) without setting ``OPENAI_API_KEY`` in CI.
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups=task_groups,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            allow_missing_judge=True,
        )


@pytest.mark.parametrize("local,command", [(False, "sbatch"), (True, "bash")])
def test_dry_run_command_quotes_a_bracketed_run_dir(tmp_path, caplog, local, command):
    # Language brackets end up in the run-dir name; the printed command must
    # still paste into a shell (unquoted, the | starts a pipe).
    with (
        patch("oellm.main._setup_logging"),  # keep caplog's handler
        patch("oellm.scheduler._setup_logging"),
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        caplog.at_level("INFO"),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups="sib200-eu[deu_Latn|fra_Latn]",
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            local=local,
        )
    (script,) = tmp_path.glob("*/submit_evals.sbatch")
    assert "[deu_latn|fra_latn]" in script.parent.name
    assert f"{command} {shlex.quote(str(script))}" in caplog.text


def test_schedule_evals_slurm_template_var_overrides(tmp_path):
    """Verify --slurm_template_var JSON overrides appear in the generated sbatch."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
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
            slurm_template_var='{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:15:00","GPUS_PER_NODE":2,"SLURM_MEM":"123G"}',
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --partition=dev-g" in sbatch_content
    assert "#SBATCH --account=myproject" in sbatch_content
    assert "#SBATCH --time=02:15:00" in sbatch_content
    assert "#SBATCH --gres=gpu:2" in sbatch_content
    # SLURM_MEM is not modeled by SlurmOverrides — it must survive the
    # EvalConfig round-trip via the extra_template_vars passthrough.
    assert "#SBATCH --mem=123G" in sbatch_content


def test_schedule_evals_nodelist(tmp_path):
    """Verify --nodelist adds an #SBATCH --nodelist directive to the sbatch."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
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
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
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


def test_schedule_evals_nodes(tmp_path):
    """When NODES is set (e.g. via clusters.yaml), the #SBATCH --nodes directive is
    kept and substituted."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {**env, "EVAL_OUTPUT_DIR": str(tmp_path), "NODES": "1"},
            clear=True,
        ),
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
    assert "#SBATCH --nodes=1" in sbatch_content


def test_schedule_evals_no_nodes(tmp_path):
    """Without NODES set the #SBATCH --nodes directive is stripped.

    Regression: template.sbatch ships ``#SBATCH --nodes=$NODES``; on clusters that
    don't define NODES (e.g. leonardo) it must not leak an unresolved ``$NODES``,
    which SLURM rejects.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("NODES", "NODELIST")}
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
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
    assert "--nodes=" not in sbatch_content
    assert "$NODES" not in sbatch_content


def test_schedule_evals_passes_bundled_lighteval_custom_tasks(tmp_path):
    """lighteval runs load the bundled custom tasks file, which is also bound
    into the container."""
    env = {k: v for k, v in os.environ.items() if k != "LIGHTEVAL_CUSTOM_TASKS"}
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {**env, "EVAL_OUTPUT_DIR": str(tmp_path), "HF_HOME": str(tmp_path / "hf")},
            clear=True,
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups="gpqa-diamond[fin_Latn]",
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )

    bundled = (
        files("oellm.resources") / "custom_lighteval_tasks" / "reasoning" / "reasoning.py"
    )
    sbatch_content = next(tmp_path.glob("**/submit_evals.sbatch")).read_text()
    assert Path(str(bundled)).is_file()
    assert f'LIGHTEVAL_CUSTOM_TASKS="{Path(str(bundled)).resolve()}"' in sbatch_content
    assert '--custom-tasks "$LIGHTEVAL_CUSTOM_TASKS"' in sbatch_content
    assert ",$LIGHTEVAL_CUSTOM_TASKS:$LIGHTEVAL_CUSTOM_TASKS" in sbatch_content


def test_schedule_evals_slurm_template_var_invalid_json(tmp_path):
    """Verify invalid slurm_template_var raises ValueError."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
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


def test_schedule_evals_dry_run_with_full_queue(tmp_path):
    """A full queue must not crash a --dry-run (ZeroDivisionError regression)."""
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=5),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "QUEUE_LIMIT": "5"}),
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
    assert "#SBATCH --array=0-0%" in sbatch_files[0].read_text()


def test_schedule_evals_rejects_comma_in_model_path(tmp_path):
    """Comma-bearing fields would be shredded by the bash CSV reader — refuse early."""
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "model_path,task_path,n_shot,eval_suite\n"
        '"EleutherAI/pythia-160m,revision=step100000",hellaswag,0,lm_eval\n'
    )
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        with pytest.raises(ValueError, match="comma"):
            schedule_evals(
                eval_csv_path=str(csv_path),
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )


def test_schedule_evals_sbatch_failure_exits_nonzero(tmp_path):
    """sbatch submission failure must surface as a non-zero exit, not exit 0."""
    import subprocess as _subprocess

    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        patch(
            "oellm.scheduler.subprocess.run",
            side_effect=FileNotFoundError("sbatch"),
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=False,
            )
        assert excinfo.value.code == 1

    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        patch(
            "oellm.scheduler.subprocess.run",
            side_effect=_subprocess.CalledProcessError(1, ["sbatch"]),
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=False,
            )
        assert excinfo.value.code == 1


def test_schedule_evals_local_failure_exits_nonzero(tmp_path):
    """A failed --local run must propagate the script's exit code."""
    import subprocess as _subprocess

    with (
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        patch(
            "oellm.scheduler.subprocess.run",
            side_effect=_subprocess.CalledProcessError(7, ["bash"]),
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                local=True,
                dry_run=False,
            )
        assert excinfo.value.code == 7


def test_download_only_prepares_packaged_judgearena_task(schedule_env):
    with (
        patch("oellm.scheduler._ensure_runtime_environment"),
        patch("oellm.envcheck.check_scheduled_environment"),
        patch("oellm.scheduler._process_model_paths") as models,
        patch("oellm.scheduler._download_judgearena_tasks") as download,
    ):
        schedule_evals(
            models="local-model",
            tasks="arena-hard-v2.0",
            n_shot=0,
            judgearena_kwargs={
                "judge.model": "VLLM/Qwen/Qwen3-4B",
                "model.baseline": "VLLM/EleutherAI/pythia-410m",
            },
            download_only=True,
            venv_path=sys.prefix,
        )

    assert download.call_args.args[0] == ["arena-hard-v2.0"]
    assert [call.args[0] for call in models.call_args_list] == [
        {"local-model"},
        {"Qwen/Qwen3-4B", "EleutherAI/pythia-410m"},
    ]


@pytest.mark.parametrize(
    "task,judge_model,skip_checks",
    [
        ("arena-hard-v0.1", "OpenRouter/google/gemma-4-31b-it", False),
        ("arena-hard-v0.1", "VLLM/Qwen/Qwen3-4B", True),
        ("unregistered-task", "VLLM/Qwen/Qwen3-4B", False),
    ],
)
def test_schedule_does_not_download_other_judges(
    schedule_env, task, judge_model, skip_checks
):
    with (
        patch("oellm.scheduler._ensure_runtime_environment"),
        patch("oellm.envcheck.check_scheduled_environment"),
        patch("oellm.scheduler._process_model_paths") as models,
        patch("oellm.scheduler._download_judgearena_tasks"),
    ):
        schedule_evals(
            models="local-model",
            tasks=task,
            n_shot=0,
            judgearena_kwargs={
                "judge.model": judge_model,
                "model.baseline": "gpt-4-0314",
            },
            download_only=True,
            skip_checks=skip_checks,
            venv_path=sys.prefix,
        )

    assert models.call_count == (0 if skip_checks else 1)


def test_schedule_forwards_judgearena_options(schedule_env, tmp_path):
    with patch("oellm.scheduler._num_jobs_in_queue", return_value=0):
        schedule_evals(
            models="local-model",
            tasks="arena-hard-v2.0",
            n_shot=0,
            judgearena_kwargs={
                "judge.prompt_preset": "alpaca-eval",
                "judge.engine_kwargs": {"enforce_eager": True},
            },
            skip_checks=True,
            venv_path=sys.prefix,
            dry_run=True,
        )

    script = next(tmp_path.glob("**/submit_evals.sbatch")).read_text()
    assert "--judge.prompt_preset alpaca-eval" in script
    assert "--judge.engine_kwargs '{\"enforce_eager\": true}'" in script


def test_schedule_rejects_task_override(schedule_env):
    with pytest.raises(ValueError, match="task"):
        schedule_evals(
            models="local-model",
            tasks="arena-hard-v2.0",
            n_shot=0,
            judgearena_kwargs={"task": "mt-bench"},
            skip_checks=True,
            venv_path=sys.prefix,
            dry_run=True,
        )
