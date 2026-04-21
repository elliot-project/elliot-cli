"""Tests for the AudioBench contrib benchmark integration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    get_all_task_group_names,
)

SUITE = "audiobench"
TOP_GROUP = "audio-audiobench"
ASR_GROUP = "audio-audiobench-asr"
ST_GROUP = "audio-audiobench-st"
REASONING_GROUP = "audio-audiobench-reasoning"

# Canonical task names that MUST be in the registry.  Assertions that
# reference individual task names go here so the audit table in the plan is
# reflected 1:1 in tests; a silent rename breaks the build.
BUCKET_B_TASKS = {
    # ASR (9)
    "audiobench_aishell_asr_zh_test",
    "audiobench_earnings21_test",
    "audiobench_earnings22_test",
    "audiobench_tedlium3_long_form_test",
    "audiobench_gigaspeech2_thai",
    "audiobench_gigaspeech2_indo",
    "audiobench_gigaspeech2_viet",
    "audiobench_seame_dev_man",
    "audiobench_seame_dev_sge",
    # ST (5)
    "audiobench_covost2_en_id_test",
    "audiobench_covost2_en_ta_test",
    "audiobench_covost2_id_en_test",
    "audiobench_covost2_zh_en_test",
    "audiobench_covost2_ta_en_test",
    # Reasoning (6)
    "audiobench_spoken_mqa_short_digit",
    "audiobench_spoken_mqa_long_digit",
    "audiobench_spoken_mqa_single_step_reasoning",
    "audiobench_spoken_mqa_multi_step_reasoning",
    "audiobench_mmau_mini",
    "audiobench_audiocaps_test",
}

BUCKET_A_DUAL = {
    "audiobench_librispeech_test_clean",
    "audiobench_librispeech_test_other",
    "audiobench_common_voice_15_en_test",
    "audiobench_gigaspeech_test",
    "audiobench_peoples_speech_test",
    "audiobench_tedlium3_test",
    "audiobench_covost2_en_zh_test",
}

ALL_PHASE1_TASKS = BUCKET_B_TASKS | BUCKET_A_DUAL


# ---------------------------------------------------------------------------
# Registry — task.py
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_registry_has_exactly_27_tasks(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        assert len(AUDIOBENCH_TASKS) == 27

    def test_registry_covers_all_phase1_task_names(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        names = {t.name for t in AUDIOBENCH_TASKS}
        assert names == ALL_PHASE1_TASKS

    def test_every_task_has_audiobench_prefix(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        for t in AUDIOBENCH_TASKS:
            assert t.name.startswith("audiobench_"), t.name

    def test_every_task_has_audiollms_or_amao_hf_repo(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        # Tasks live on AudioLLMs/* with the single exception of spoken-mqa
        # (amao0o0/spoken-mqa).
        for t in AUDIOBENCH_TASKS:
            assert t.hf_repo.startswith(("AudioLLMs/", "amao0o0/")), (
                f"{t.name} has unexpected repo {t.hf_repo}"
            )

    def test_asr_tasks_all_use_wer(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        for t in AUDIOBENCH_TASKS:
            if t.family == "asr":
                assert t.metric == "wer", f"{t.name}: {t.metric}"

    def test_st_tasks_all_use_bleu(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        for t in AUDIOBENCH_TASKS:
            if t.family == "st":
                assert t.metric == "bleu", f"{t.name}: {t.metric}"

    def test_gigaspeech2_tasks_share_repo_and_differ_by_data_dir(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        gs2 = [t for t in AUDIOBENCH_TASKS if "gigaspeech2" in t.name]
        assert len(gs2) == 3
        # All share the same HF repo.
        assert {t.hf_repo for t in gs2} == {"AudioLLMs/gigaspeech2-test"}
        # Each has a distinct data_dir.
        assert {t.data_dir for t in gs2} == {"th-test", "id-test", "vi-test"}

    def test_spoken_mqa_tasks_share_repo_and_differ_by_data_dir(self):
        from oellm.contrib.audiobench.task import AUDIOBENCH_TASKS

        smqa = [t for t in AUDIOBENCH_TASKS if "spoken_mqa" in t.name]
        assert len(smqa) == 4
        assert {t.hf_repo for t in smqa} == {"amao0o0/spoken-mqa"}
        assert {t.data_dir for t in smqa} == {
            "short_digit",
            "long_digit",
            "single_step_reasoning",
            "multi_step_reasoning",
        }

    def test_get_task_spec_returns_spec(self):
        from oellm.contrib.audiobench.task import get_task_spec

        spec = get_task_spec("audiobench_librispeech_test_clean")
        assert spec.upstream_name == "librispeech_test_clean"
        assert spec.metric == "wer"
        assert spec.family == "asr"

    def test_get_task_spec_unknown_raises(self):
        from oellm.contrib.audiobench.task import get_task_spec

        with pytest.raises(KeyError, match="Unknown AudioBench task"):
            get_task_spec("audiobench_does_not_exist")


# ---------------------------------------------------------------------------
# Adapter — adapter.py
# ---------------------------------------------------------------------------


class TestAudioBenchModelAdapter:
    @pytest.fixture
    def adapter_cls(self):
        from oellm.contrib.audiobench.adapter import AudioBenchModelAdapter
        from oellm.core.base_model_adapter import BaseModelAdapter

        return AudioBenchModelAdapter, BaseModelAdapter

    def test_is_base_model_adapter(self, adapter_cls):
        cls, base = adapter_cls
        assert issubclass(cls, base)

    def test_qwen2_audio(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen2-Audio-7B-Instruct").to_contrib_flags() == "qwen2_audio"

    def test_qwen_audio(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen-Audio-Chat").to_contrib_flags() == "qwen2_audio"

    def test_salmonn(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("tsinghua/SALMONN-13B").to_contrib_flags() == "salmonn"

    def test_ltu(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("MIT/ltu-as").to_contrib_flags() == "ltu"

    def test_whisper(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("openai/whisper-large-v3").to_contrib_flags() == "whisper"

    def test_audio_flamingo(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("nvidia/audio-flamingo-2").to_contrib_flags() == "audioflamingo"

    def test_meralion(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Singapore-NLP/MERaLiON-7B").to_contrib_flags() == "meralion"

    def test_unknown_defaults_to_generic(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("random/unknown-model").to_contrib_flags() == "generic"

    def test_module_level_detect_function(self):
        from oellm.contrib.audiobench.adapter import detect_audiobench_model_type

        assert detect_audiobench_model_type("Qwen/Qwen2-Audio-7B") == "qwen2_audio"
        assert detect_audiobench_model_type("completely/unknown") == "generic"


# ---------------------------------------------------------------------------
# Suite plugin protocol — suite.py
# ---------------------------------------------------------------------------


class TestSuiteProtocol:
    @pytest.fixture
    def suite(self):
        import oellm.contrib.audiobench.suite as s

        return s

    def test_suite_name(self, suite):
        assert suite.SUITE_NAME == "audiobench"

    def test_cluster_env_vars_declared(self, suite):
        assert "AUDIOBENCH_DIR" in suite.CLUSTER_ENV_VARS

    def test_task_groups_contains_all_four_groups(self, suite):
        groups = suite.TASK_GROUPS["task_groups"]
        for g in (TOP_GROUP, ASR_GROUP, ST_GROUP, REASONING_GROUP):
            assert g in groups, f"{g} missing from TASK_GROUPS"

    def test_top_level_group_has_all_27_tasks(self, suite):
        tasks = suite.TASK_GROUPS["task_groups"][TOP_GROUP]["tasks"]
        assert len(tasks) == 27

    def test_task_metrics_present_for_all_leaves(self, suite):
        metrics = suite.TASK_GROUPS["task_metrics"]
        assert set(metrics.keys()) == ALL_PHASE1_TASKS

    def test_all_groups_are_zero_shot(self, suite):
        for name in (TOP_GROUP, ASR_GROUP, ST_GROUP, REASONING_GROUP):
            group = suite.TASK_GROUPS["task_groups"][name]
            assert group["n_shots"] == [0]
            assert group["suite"] == SUITE

    def test_detect_model_flags_qwen2_audio(self, suite):
        assert suite.detect_model_flags("Qwen/Qwen2-Audio-7B-Instruct") == "qwen2_audio"

    def test_detect_model_flags_unknown_defaults_to_generic(self, suite):
        assert suite.detect_model_flags("some/unknown-model") == "generic"

    def test_parse_results_recognises_audiobench_json(self, suite):
        data = {
            "model_name_or_path": "/path/to/model",
            "results": {
                "audiobench_librispeech_test_clean": {"wer": 0.047},
            },
            "configs": {"audiobench_librispeech_test_clean": {"num_fewshot": 0}},
        }
        result = suite.parse_results(data)
        assert result is not None
        model_id, task_name, n_shot, metrics = result
        assert model_id == "/path/to/model"
        assert task_name == "audiobench_librispeech_test_clean"
        assert n_shot == 0
        assert metrics["wer"] == pytest.approx(0.047)

    def test_parse_results_rejects_non_audiobench_json(self, suite):
        # lmms-eval style — no audiobench_ prefix.
        data = {
            "model_name_or_path": "some/model",
            "results": {"librispeech_test_clean": {"wer,none": 0.05}},
            "configs": {"librispeech_test_clean": {"num_fewshot": 0}},
        }
        assert suite.parse_results(data) is None

    def test_parse_results_empty_returns_none(self, suite):
        assert suite.parse_results({}) is None

    def test_parse_results_malformed_returns_none(self, suite):
        assert suite.parse_results({"results": "not a dict"}) is None


# ---------------------------------------------------------------------------
# TASK_GROUPS integration with core registry.
# ---------------------------------------------------------------------------


class TestTaskGroupsIntegration:
    def test_groups_registered_via_registry(self):
        all_names = get_all_task_group_names()
        for g in (TOP_GROUP, ASR_GROUP, ST_GROUP, REASONING_GROUP):
            assert g in all_names

    def test_top_group_expands_to_27_zero_shot_tasks(self):
        results = _expand_task_groups([TOP_GROUP])
        assert len(results) == 27
        for r in results:
            assert r.n_shot == 0
            assert r.suite == SUITE

    def test_top_group_expands_to_expected_task_names(self):
        results = _expand_task_groups([TOP_GROUP])
        assert {r.task for r in results} == ALL_PHASE1_TASKS

    def test_asr_group_has_15_leaves(self):
        results = _expand_task_groups([ASR_GROUP])
        # 9 bucket-B ASR + 6 bucket-A dual ASR = 15.
        assert len(results) == 15
        for r in results:
            assert r.suite == SUITE

    def test_st_group_has_6_leaves(self):
        results = _expand_task_groups([ST_GROUP])
        # 5 bucket-B ST + 1 bucket-A dual (en→zh) = 6.
        assert len(results) == 6

    def test_reasoning_group_has_6_leaves(self):
        results = _expand_task_groups([REASONING_GROUP])
        # 4 spoken-mqa + mmau_mini + audiocaps = 6.
        assert len(results) == 6

    def test_dataset_specs_flag_snapshot_download(self):
        # Auto-derived from the ``audio-*`` group-name prefix in
        # _collect_dataset_specs.
        specs = _collect_dataset_specs([TOP_GROUP])
        assert specs, "No dataset specs returned"
        for s in specs:
            assert s.needs_snapshot_download, (
                f"DatasetSpec for {s.repo_id} missing needs_snapshot_download=True"
            )

    def test_dataset_specs_dedupe_shared_repos(self):
        # gigaspeech2 (3 tasks) → 1 spec; spoken-mqa (4 tasks) → 1 spec.
        specs = _collect_dataset_specs([TOP_GROUP])
        repo_ids = [s.repo_id for s in specs]
        assert repo_ids.count("AudioLLMs/gigaspeech2-test") == 1
        assert repo_ids.count("amao0o0/spoken-mqa") == 1

    def test_dataset_specs_contain_audiollms_repos(self):
        specs = _collect_dataset_specs([TOP_GROUP])
        repo_ids = {s.repo_id for s in specs}
        # Sanity-check a handful of expected entries.
        assert "AudioLLMs/librispeech_test_clean" in repo_ids
        assert "AudioLLMs/earnings21_test" in repo_ids
        assert "AudioLLMs/MMAU-mini" in repo_ids
        assert "amao0o0/spoken-mqa" in repo_ids


# ---------------------------------------------------------------------------
# Registry auto-discovery.
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    def test_audiobench_suite_is_auto_discovered(self):
        # Clear the _discover() cache so this test doesn't rely on import
        # order from earlier tests.
        from oellm import registry

        registry._discover.cache_clear()
        mod = registry.get_suite("audiobench")
        assert mod.SUITE_NAME == "audiobench"
        assert hasattr(mod, "run")
        assert hasattr(mod, "parse_results")
        assert hasattr(mod, "detect_model_flags")

    def test_task_groups_merged_into_registry(self):
        from oellm import registry

        registry._discover.cache_clear()
        merged = registry.get_all_task_groups()
        assert TOP_GROUP in merged["task_groups"]
        # task_metrics come through too.
        assert "audiobench_librispeech_test_clean" in merged["task_metrics"]


# ---------------------------------------------------------------------------
# EvalRunner — resolve_suite wires audiobench through the adapter.
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    def test_resolve_suite_appends_model_flag(self):
        from oellm.constants import EvaluationJob
        from oellm.runner import EvalRunner

        runner = EvalRunner()
        job = EvaluationJob(
            model_path="Qwen/Qwen2-Audio-7B-Instruct",
            task_path="audiobench_librispeech_test_clean",
            n_shot=0,
            eval_suite="audiobench",
        )
        result = runner.resolve_suite(job)
        assert result == "audiobench:qwen2_audio"

    def test_resolve_suite_generic_fallback(self):
        from oellm.constants import EvaluationJob
        from oellm.runner import EvalRunner

        runner = EvalRunner()
        job = EvaluationJob(
            model_path="some/unknown-model",
            task_path="audiobench_mmau_mini",
            n_shot=0,
            eval_suite="audiobench",
        )
        result = runner.resolve_suite(job)
        assert result == "audiobench:generic"


# ---------------------------------------------------------------------------
# run() subprocess harness — exercise with a mocked subprocess.
# ---------------------------------------------------------------------------


class TestRunHarness:
    """Exercise suite.run() with a mocked subprocess, verifying the CLI
    it would invoke and the output JSON it writes.
    """

    def _fake_audiobench_tree(self, tmp_path: Path) -> Path:
        """Create a minimal directory tree that looks like an AudioBench clone."""
        ab_dir = tmp_path / "AudioBench"
        (ab_dir / "src").mkdir(parents=True)
        (ab_dir / "src" / "main_evaluate.py").write_text("# placeholder\n")
        return ab_dir

    def test_run_missing_audiobench_dir_raises(self, tmp_path):
        from oellm.contrib.audiobench.suite import run

        with pytest.raises(RuntimeError, match="AUDIOBENCH_DIR must be set"):
            run(
                model_path="Qwen/Qwen2-Audio-7B",
                task="audiobench_librispeech_test_clean",
                n_shot=0,
                output_path=tmp_path / "out.json",
                model_flags="qwen2_audio",
                env={},  # no AUDIOBENCH_DIR
            )

    def test_run_missing_entrypoint_raises(self, tmp_path):
        from oellm.contrib.audiobench.suite import run

        bad_dir = tmp_path / "not-audiobench"
        bad_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="AudioBench entry point"):
            run(
                model_path="Qwen/Qwen2-Audio-7B",
                task="audiobench_librispeech_test_clean",
                n_shot=0,
                output_path=tmp_path / "out.json",
                model_flags="qwen2_audio",
                env={"AUDIOBENCH_DIR": str(bad_dir)},
            )

    def test_run_invokes_subprocess_with_expected_cli(self, tmp_path):
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        def fake_run(cmd, cwd, env, check):
            # Write a fake AudioBench result JSON into the run_dir that
            # _extract_metrics will pick up.
            run_dir = Path(cmd[cmd.index("--log_dir") + 1])
            (run_dir / "task_result.json").write_text(json.dumps({"wer": 0.063}))
            return _FakeCompletedProcess(0)

        with patch(
            "oellm.contrib.audiobench.suite.subprocess.run", side_effect=fake_run
        ) as mock_sp:
            suite.run(
                model_path="Qwen/Qwen2-Audio-7B-Instruct",
                task="audiobench_librispeech_test_clean",
                n_shot=0,
                output_path=output_path,
                model_flags="qwen2_audio",
                env={"AUDIOBENCH_DIR": str(ab_dir), "LIMIT": "100"},
            )

        assert mock_sp.call_count == 1
        cmd = mock_sp.call_args.args[0]
        assert cmd[:2] == ["python", "src/main_evaluate.py"]
        assert "--dataset" in cmd
        assert cmd[cmd.index("--dataset") + 1] == "librispeech_test_clean"
        assert cmd[cmd.index("--model") + 1] == "qwen2_audio"
        assert cmd[cmd.index("--model_name") + 1] == "Qwen/Qwen2-Audio-7B-Instruct"
        assert cmd[cmd.index("--metrics") + 1] == "wer"
        # LIMIT propagated.
        assert cmd[cmd.index("--number_of_samples") + 1] == "100"
        # cwd is AUDIOBENCH_DIR.
        assert mock_sp.call_args.kwargs["cwd"] == str(ab_dir)

        # Output JSON is lmms-eval-shaped and contains the extracted metric.
        body = json.loads(output_path.read_text())
        assert body["model_name_or_path"] == "Qwen/Qwen2-Audio-7B-Instruct"
        assert body["results"]["audiobench_librispeech_test_clean"][
            "wer"
        ] == pytest.approx(0.063)
        assert body["configs"]["audiobench_librispeech_test_clean"]["num_fewshot"] == 0

    def test_run_forwards_data_dir_for_gigaspeech2(self, tmp_path):
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        def fake_run(cmd, cwd, env, check):
            run_dir = Path(cmd[cmd.index("--log_dir") + 1])
            (run_dir / "gs2.json").write_text(json.dumps({"wer": 0.12}))
            return _FakeCompletedProcess(0)

        with patch(
            "oellm.contrib.audiobench.suite.subprocess.run", side_effect=fake_run
        ) as mock_sp:
            suite.run(
                model_path="Qwen/Qwen2-Audio-7B",
                task="audiobench_gigaspeech2_thai",
                n_shot=0,
                output_path=output_path,
                model_flags="qwen2_audio",
                env={"AUDIOBENCH_DIR": str(ab_dir)},
            )

        cmd = mock_sp.call_args.args[0]
        assert "--data_dir" in cmd
        assert cmd[cmd.index("--data_dir") + 1] == "th-test"

    def test_run_omits_number_of_samples_when_limit_empty(self, tmp_path):
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        def fake_run(cmd, cwd, env, check):
            run_dir = Path(cmd[cmd.index("--log_dir") + 1])
            (run_dir / "r.json").write_text(json.dumps({"wer": 0.1}))
            return _FakeCompletedProcess(0)

        with patch(
            "oellm.contrib.audiobench.suite.subprocess.run", side_effect=fake_run
        ) as mock_sp:
            suite.run(
                model_path="Qwen/Qwen2-Audio-7B",
                task="audiobench_librispeech_test_clean",
                n_shot=0,
                output_path=output_path,
                model_flags="qwen2_audio",
                env={"AUDIOBENCH_DIR": str(ab_dir), "LIMIT": ""},
            )

        cmd = mock_sp.call_args.args[0]
        assert "--number_of_samples" not in cmd

    def test_run_nonzero_exit_raises(self, tmp_path):
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        with patch(
            "oellm.contrib.audiobench.suite.subprocess.run",
            return_value=_FakeCompletedProcess(1),
        ):
            with pytest.raises(RuntimeError, match="AudioBench exited with code 1"):
                suite.run(
                    model_path="Qwen/Qwen2-Audio-7B",
                    task="audiobench_librispeech_test_clean",
                    n_shot=0,
                    output_path=output_path,
                    model_flags="qwen2_audio",
                    env={"AUDIOBENCH_DIR": str(ab_dir)},
                )

    def test_run_handles_nested_metric_json(self, tmp_path):
        """AudioBench output format has drifted; support
        ``{"metrics": {"wer": {"score": 0.05}}}`` as well as flat
        ``{"wer": 0.05}``.
        """
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        def fake_run(cmd, cwd, env, check):
            run_dir = Path(cmd[cmd.index("--log_dir") + 1])
            (run_dir / "nested.json").write_text(
                json.dumps({"metrics": {"wer": {"score": 0.051, "notes": "nested"}}})
            )
            return _FakeCompletedProcess(0)

        with patch("oellm.contrib.audiobench.suite.subprocess.run", side_effect=fake_run):
            suite.run(
                model_path="Qwen/Qwen2-Audio-7B",
                task="audiobench_librispeech_test_clean",
                n_shot=0,
                output_path=output_path,
                model_flags="qwen2_audio",
                env={"AUDIOBENCH_DIR": str(ab_dir)},
            )

        body = json.loads(output_path.read_text())
        assert body["results"]["audiobench_librispeech_test_clean"][
            "wer"
        ] == pytest.approx(0.051)

    def test_run_missing_metric_in_output_raises(self, tmp_path):
        from oellm.contrib.audiobench import suite

        ab_dir = self._fake_audiobench_tree(tmp_path)
        output_path = tmp_path / "result.json"

        def fake_run(cmd, cwd, env, check):
            run_dir = Path(cmd[cmd.index("--log_dir") + 1])
            (run_dir / "no_metric.json").write_text(json.dumps({"irrelevant": 1}))
            return _FakeCompletedProcess(0)

        with patch("oellm.contrib.audiobench.suite.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="Could not locate metric"):
                suite.run(
                    model_path="Qwen/Qwen2-Audio-7B",
                    task="audiobench_librispeech_test_clean",
                    n_shot=0,
                    output_path=output_path,
                    model_flags="qwen2_audio",
                    env={"AUDIOBENCH_DIR": str(ab_dir)},
                )


class _FakeCompletedProcess:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


# ---------------------------------------------------------------------------
# schedule_evals dry-run — wiring smoke test.
# ---------------------------------------------------------------------------


class TestScheduleEvalsDryRun:
    def test_dry_run_writes_audiobench_suite_to_csv(self, tmp_path):
        import pandas as pd

        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="Qwen/Qwen2-Audio-7B-Instruct",
                task_groups=ASR_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        # All rows route to audiobench (with or without model-flag suffix).
        assert all(s.startswith("audiobench") for s in df["eval_suite"].unique())
        # task_path column contains canonical audiobench_ names.
        assert all(t.startswith("audiobench_") for t in df["task_path"].unique())

    def test_dry_run_sbatch_contains_contrib_dispatch(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="Qwen/Qwen2-Audio-7B-Instruct",
                task_groups=TOP_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch_files) == 1
        content = sbatch_files[0].read_text()
        assert "oellm.contrib.dispatch" in content
        # LIMIT is now exported so contrib plugins can read it.
        assert "export LIMIT=" in content


# ---------------------------------------------------------------------------
# collect_results compatibility — verify a run() output flows through unchanged.
# ---------------------------------------------------------------------------


class TestCollectResultsCompat:
    def test_collect_results_parses_audiobench_json(self, tmp_path):
        import pandas as pd

        from oellm.main import collect_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        mock_output = {
            "model_name_or_path": "/cluster/models/Qwen2-Audio-7B",
            "results": {
                "audiobench_librispeech_test_clean": {"wer": 0.052},
            },
            "configs": {"audiobench_librispeech_test_clean": {"num_fewshot": 0}},
        }
        (results_dir / "ab123.json").write_text(json.dumps(mock_output))

        output_csv = str(tmp_path / "results.csv")
        collect_results(str(tmp_path), output_csv=output_csv)

        df = pd.read_csv(output_csv)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == "audiobench_librispeech_test_clean"
        assert float(row["performance"]) == pytest.approx(0.052)
        assert row["model_name"] == "/cluster/models/Qwen2-Audio-7B"
