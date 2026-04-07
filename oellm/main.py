import logging
from pathlib import Path

import typer
from typer import rich_utils

from oellm.config import EvalConfig
from oellm.results import collect_results
from oellm.utils import _filter_warnings, _setup_logging

# Override Typer's default cyan colour scheme with colours that are readable
# on both light (white) and dark terminal backgrounds.
rich_utils.COLOR_OPTIONS_PANEL_TITLE = "bold blue"
rich_utils.COLOR_ARGUMENTS_PANEL_TITLE = "bold blue"
rich_utils.COLOR_COMMANDS_PANEL_TITLE = "bold blue"
rich_utils.STYLE_OPTION = "bold blue"
rich_utils.STYLE_SWITCH = "bold dark_green"
rich_utils.STYLE_NEGATIVE_OPTION = "bold magenta"
rich_utils.STYLE_NEGATIVE_SWITCH = "bold magenta"
rich_utils.STYLE_METAVAR = "dark_orange3"
rich_utils.STYLE_OPTION_DEFAULT = "dim"

app = typer.Typer(
    name="oellm",
    help="ELLIOT: Multi-cluster evaluation tool for language models",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    config: str | None = None,
    max_array_len: int = 128,
    limit: int | None = None,
    verbose: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    skip_checks: bool = False,
    trust_remote_code: bool = True,
    venv_path: str | None = None,
    slurm_template_var: str | None = None,
) -> None:
    """Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: Comma-separated model paths or Hugging Face model identifiers.
            For local paths, directories are expanded to discover all checkpoints.
            Cannot be combined with eval_csv_path.
        tasks: Comma-separated task names (lm_eval) or paths. Requires n_shot.
        task_groups: Comma-separated task group names from task-groups.yaml.
            Expands into (task, n_shots, suite) tuples; n_shot is ignored.
        n_shot: Number(s) of shots applied to tasks.
        eval_csv_path: Path to a CSV with columns model_path, task_path, n_shot[, eval_suite].
            Exclusive with models/tasks/task_groups/n_shot.
        config: Path to a YAML config file. CLI flags override YAML values.
        max_array_len: Maximum concurrent SLURM array jobs.
        limit: Limit samples per task (passes --limit / --max_samples to the engine).
        download_only: Only pre-download models and datasets, then exit.
        dry_run: Generate the SLURM script without submitting.
        skip_checks: Skip container/model/dataset validation for faster iteration.
        trust_remote_code: Trust remote code when downloading datasets.
        venv_path: Python venv path. When set, runs in venv instead of Singularity.
        slurm_template_var: JSON object of SLURM overrides, e.g.
            '{"PARTITION":"dev-g","ACCOUNT":"FOO","TIME":"02:00:00","GPUS_PER_NODE":2}'
    """
    from oellm.scheduler import schedule_evals as _sched

    cli_cfg = EvalConfig.from_cli_kwargs(
        models=models,
        tasks=tasks,
        task_groups=task_groups,
        n_shot=n_shot,
        eval_csv_path=eval_csv_path,
        max_array_len=max_array_len,
        limit=limit,
        verbose=verbose,
        download_only=download_only,
        dry_run=dry_run,
        skip_checks=skip_checks,
        trust_remote_code=trust_remote_code,
        venv_path=venv_path,
        slurm_template_var=slurm_template_var,
    )

    if config:
        yaml_cfg = EvalConfig.from_yaml(config)
        cfg = yaml_cfg.merge(cli_cfg)
        logging.info(f"Loaded config from {config} (CLI flags override)")
    else:
        cfg = cli_cfg

    cfg.validate()
    _setup_logging(cfg.verbose)

    models_str: str | None = None
    if cfg._model_paths():
        models_str = ",".join(cfg._model_paths())

    _sched(
        models=models_str,
        tasks=",".join(cfg.tasks) if cfg.tasks else None,
        task_groups=",".join(cfg.task_groups) if cfg.task_groups else None,
        n_shot=cfg.n_shot,
        eval_csv_path=cfg.eval_csv_path,
        max_array_len=cfg.slurm.max_array_len,
        limit=cfg.limit,
        verbose=cfg.verbose,
        download_only=cfg.download_only,
        dry_run=cfg.dry_run,
        skip_checks=cfg.skip_checks,
        trust_remote_code=cfg.trust_remote_code,
        venv_path=cfg.venv_path,
        slurm_template_var=cfg.slurm_template_var_json,
    )


def list_tasks(*, group: str | None = None) -> None:
    """List available task groups and their tasks.

    Args:
        group: If provided, show tasks within this specific group.
    """
    from rich.table import Table

    from oellm.task_groups import TaskGroup, _parse_task_groups, get_all_task_group_names
    from oellm.utils import get_console

    console = get_console()
    all_names = get_all_task_group_names()

    if group:
        all_names = [group]

    parsed = _parse_task_groups(all_names)

    table = Table(title="Available Task Groups")
    table.add_column("Group", style="bold")
    table.add_column("Suite")
    table.add_column("Tasks", justify="right")
    table.add_column("N-shots")
    table.add_column("Description")

    for name in sorted(parsed.keys()):
        g = parsed[name]
        if isinstance(g, TaskGroup):
            n_shots_set = set()
            for t in g.tasks:
                for s in t.n_shots or []:
                    n_shots_set.add(s)
            n_shots_str = ", ".join(str(s) for s in sorted(n_shots_set))
            table.add_row(
                name,
                g.suite,
                str(len(g.tasks)),
                n_shots_str,
                g.description,
            )
        else:
            # SuperGroup
            total_tasks = sum(len(sg.tasks) for sg in g.task_groups)
            table.add_row(
                name,
                "mixed",
                str(total_tasks),
                "",
                g.description,
            )

    console.print(table)


def compare(
    result_a: str,
    result_b: str,
    *,
    verbose: bool = False,
) -> None:
    """Compare two evaluation result files or directories.

    Args:
        result_a: Path to first results JSON file or directory containing results.json
        result_b: Path to second results JSON file or directory containing results.json
        verbose: Enable verbose logging
    """
    import json

    from rich.table import Table

    from oellm.utils import get_console

    _setup_logging(verbose)

    def _load_results(path_str: str) -> list[dict]:
        p = Path(path_str)
        if p.is_dir():
            p = p / "results.json"
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {p}")
        data = json.loads(p.read_text())
        return data.get("results", [])

    results_a = _load_results(result_a)
    results_b = _load_results(result_b)

    # Index by (task, n_shot, metric)
    def _index(results: list[dict]) -> dict[tuple, float]:
        idx = {}
        for r in results:
            key = (r.get("task", ""), r.get("n_shot", 0), r.get("metric", ""))
            idx[key] = r.get("performance", 0.0)
        return idx

    idx_a = _index(results_a)
    idx_b = _index(results_b)
    all_keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))

    console = get_console()
    table = Table(title="Comparison")
    table.add_column("Task", style="bold")
    table.add_column("N-shot", justify="right")
    table.add_column("Metric")
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("\u0394", justify="right")  # Delta

    for task, n_shot, metric in all_keys:
        val_a = idx_a.get((task, n_shot, metric))
        val_b = idx_b.get((task, n_shot, metric))
        str_a = f"{val_a:.4f}" if val_a is not None else "\u2014"
        str_b = f"{val_b:.4f}" if val_b is not None else "\u2014"
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            str_delta = f"{delta:+.4f}"
        else:
            str_delta = "\u2014"
        table.add_row(task, str(n_shot), metric, str_a, str_b, str_delta)

    console.print(table)


def eval_command(
    config: str | None = None,
    *,
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: list[int] | None = None,
    eval_csv_path: str | None = None,
    max_array_len: int = 128,
    limit: int | None = None,
    verbose: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    skip_checks: bool = False,
    trust_remote_code: bool = True,
    venv_path: str | None = None,
    slurm_template_var: str | None = None,
) -> None:
    """Run evaluations from a YAML config file.

    All CLI flags override values in --config. Delegates to schedule-eval.

    Args:
        config: Path to a YAML config file.
        models: Comma-separated model paths or HF identifiers (overrides config).
        tasks: Comma-separated task names (overrides config).
        task_groups: Comma-separated task group names (overrides config).
        n_shot: Number(s) of shots applied to tasks (overrides config).
        eval_csv_path: Path to a CSV with evaluation jobs (overrides config).
        max_array_len: Maximum concurrent SLURM array jobs.
        limit: Limit samples per task.
        download_only: Only pre-download models and datasets, then exit.
        dry_run: Generate the SLURM script without submitting.
        skip_checks: Skip container/model/dataset validation.
        trust_remote_code: Trust remote code when downloading datasets.
        venv_path: Python venv path. When set, runs in venv instead of Singularity.
        slurm_template_var: JSON object of SLURM overrides.
        verbose: Enable verbose logging.
    """
    schedule_evals(
        models=models,
        tasks=tasks,
        task_groups=task_groups,
        n_shot=n_shot,
        eval_csv_path=eval_csv_path,
        config=config,
        max_array_len=max_array_len,
        limit=limit,
        verbose=verbose,
        download_only=download_only,
        dry_run=dry_run,
        skip_checks=skip_checks,
        trust_remote_code=trust_remote_code,
        venv_path=venv_path,
        slurm_template_var=slurm_template_var,
    )


# Register CLI commands
app.command("schedule-eval")(schedule_evals)
app.command("eval")(eval_command)
app.command("collect-results")(collect_results)
app.command("list-tasks")(list_tasks)
app.command("compare")(compare)


def main() -> None:
    _filter_warnings()
    app()
