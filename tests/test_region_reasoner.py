"""Tests for the RegionReasoner contrib benchmark integration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.core.base_metric import BaseMetric
from oellm.core.base_task import BaseTask
from oellm.task_groups import (
    _collect_dataset_specs,
    _collect_hf_dataset_files,
    _expand_task_groups,
    get_all_task_group_names,
)

RR_TASK_GROUP = "region-reasoner"
RR_TASK_NAME = "regionreasoner_refcocog"
RR_TEST_DATA_REPO = "lmsdss/regionreasoner_test_data"


# ---------------------------------------------------------------------------
# Task group integration (end-to-end: registry → task_groups merge → YAML)
# ---------------------------------------------------------------------------


class TestRegionReasonerTaskGroup:
    def test_task_group_present_in_all_names(self):
        assert RR_TASK_GROUP in get_all_task_group_names()

    def test_task_group_expands_to_correct_task(self):
        results = _expand_task_groups([RR_TASK_GROUP])
        task_names = {r.task for r in results}
        assert RR_TASK_NAME in task_names

    def test_task_group_suite_is_region_reasoner(self):
        results = _expand_task_groups([RR_TASK_GROUP])
        for r in results:
            assert r.suite == "region_reasoner", (
                f"Expected suite 'region_reasoner', got '{r.suite}'"
            )

    def test_task_group_n_shot_is_zero(self):
        results = _expand_task_groups([RR_TASK_GROUP])
        for r in results:
            assert r.n_shot == 0

    def test_no_dataset_pre_download(self):
        # No HuggingFace dataset (load_dataset-style) is declared for this task group.
        specs = _collect_dataset_specs([RR_TASK_GROUP])
        assert specs == []

    def test_hf_dataset_files_declared(self):
        import oellm.contrib.region_reasoner.suite as s

        tasks = s.TASK_GROUPS["task_groups"][RR_TASK_GROUP]["tasks"]
        repo_ids = [
            spec["repo_id"] for task in tasks for spec in task.get("hf_dataset_files", [])
        ]
        assert RR_TEST_DATA_REPO in repo_ids

    def test_collect_hf_dataset_files_returns_correct_spec(self):
        specs = _collect_hf_dataset_files([RR_TASK_GROUP])
        assert len(specs) == 1
        assert specs[0]["repo_id"] == RR_TEST_DATA_REPO
        assert "raw/refcocog_multi_turn.json" in specs[0]["patterns"]

    def test_task_groups_generated_from_task_class(self):
        # TASK_GROUPS must be generated from RegionReasonerTask, not hardcoded.
        from oellm.contrib.region_reasoner.task import RegionReasonerTask

        generated = RegionReasonerTask.to_task_groups_dict()
        import oellm.contrib.region_reasoner.suite as s

        assert s.TASK_GROUPS == generated


# ---------------------------------------------------------------------------
# BaseTask subclass
# ---------------------------------------------------------------------------


class TestRegionReasonerTask:
    @pytest.fixture
    def task(self):
        from oellm.contrib.region_reasoner.task import RegionReasonerTask

        return RegionReasonerTask()

    def test_is_base_task_instance(self, task):
        assert isinstance(task, BaseTask)

    def test_name(self, task):
        assert task.name == RR_TASK_NAME

    def test_suite(self, task):
        assert task.suite == "region_reasoner"

    def test_n_shots(self, task):
        assert task.n_shots == [0]

    def test_dataset_specs_empty(self, task):
        # Data is accessed via hf_dataset_files (snapshot_download), not load_dataset.
        assert task.dataset_specs == []

    def test_hf_dataset_files(self, task):
        repo_ids = [f["repo_id"] for f in task.hf_dataset_files]
        assert RR_TEST_DATA_REPO in repo_ids

    def test_hf_models(self, task):
        assert "Ricky06662/TaskRouter-1.5B" in task.hf_models
        assert "facebook/sam2-hiera-large" in task.hf_models

    def test_primary_metric(self, task):
        assert task.primary_metric == "gIoU"

    def test_task_group_name(self, task):
        assert task.task_group_name == RR_TASK_GROUP

    def test_engine_task_name_defaults_to_name(self, task):
        assert task.engine_task_name == task.name

    def test_to_task_groups_dict_structure(self, task):
        d = task.to_task_groups_dict()
        assert "task_metrics" in d
        assert d["task_metrics"][RR_TASK_NAME] == "gIoU"
        assert RR_TASK_GROUP in d["task_groups"]
        tasks = d["task_groups"][RR_TASK_GROUP]["tasks"]
        assert any(t["task"] == RR_TASK_NAME for t in tasks)


# ---------------------------------------------------------------------------
# BaseMetric subclasses
# ---------------------------------------------------------------------------


def _sample(intersection: int, union: int, bbox_iou: float = 0.0) -> str:
    """Helper: JSON-serialise a sample dict for metric inputs."""
    return json.dumps(
        {
            "intersection": intersection,
            "union": union,
            "bbox_iou": bbox_iou,
        }
    )


class TestGIoU:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.region_reasoner.metrics import GIoU

        return GIoU()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "gIoU"

    def test_perfect_overlap(self, metric):
        s = _sample(100, 100)
        assert metric.compute([s], [""]) == pytest.approx(1.0)

    def test_zero_overlap(self, metric):
        s = _sample(0, 200)
        assert metric.compute([s], [""]) == pytest.approx(0.0)

    def test_partial_overlap(self, metric):
        s = _sample(25, 175)
        assert metric.compute([s], [""]) == pytest.approx(25 / 175, abs=1e-4)

    def test_mean_over_multiple_samples(self, metric):
        perfect = _sample(100, 100)
        zero = _sample(0, 200)
        score = metric.compute([perfect, zero], ["", ""])
        assert score == pytest.approx(0.5)

    def test_empty_input(self, metric):
        assert metric.compute([], []) == pytest.approx(0.0)

    def test_null_sample(self, metric):
        score = metric.compute(["null"], [""])
        assert score == pytest.approx(0.0)


class TestCIoU:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.region_reasoner.metrics import CIoU

        return CIoU()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "cIoU"

    def test_perfect_overlap(self, metric):
        s = _sample(100, 100)
        assert metric.compute([s], [""]) == pytest.approx(1.0)

    def test_zero_overlap(self, metric):
        s = _sample(0, 200)
        assert metric.compute([s], [""]) == pytest.approx(0.0)

    def test_cumulative_formula_differs_from_giou(self, metric):
        from oellm.contrib.region_reasoner.metrics import GIoU

        giou = GIoU()
        # Sample 1: IoU = 100/100 = 1.0
        # Sample 2: IoU = 50/200 = 0.25
        s1 = _sample(100, 100)
        s2 = _sample(50, 200)
        preds = [s1, s2]
        refs = ["", ""]
        ciou_val = metric.compute(preds, refs)  # (100+50)/(100+200) = 0.5
        giou_val = giou.compute(preds, refs)  # (1.0+0.25)/2 = 0.625
        assert ciou_val == pytest.approx(0.5)
        assert giou_val == pytest.approx(0.625)
        assert ciou_val != pytest.approx(giou_val)

    def test_empty_input(self, metric):
        assert metric.compute([], []) == pytest.approx(0.0)


class TestBboxAP:
    @pytest.fixture
    def metric(self):
        from oellm.contrib.region_reasoner.metrics import BboxAP

        return BboxAP()

    def test_is_base_metric(self, metric):
        assert isinstance(metric, BaseMetric)

    def test_name(self, metric):
        assert metric.name == "bbox_AP"

    def test_all_correct(self, metric):
        s = _sample(100, 100, bbox_iou=0.9)
        assert metric.compute([s, s], ["", ""]) == pytest.approx(1.0)

    def test_none_correct(self, metric):
        s = _sample(10, 200, bbox_iou=0.3)
        assert metric.compute([s], [""]) == pytest.approx(0.0)

    def test_threshold_at_half(self, metric):
        above = _sample(80, 100, bbox_iou=0.6)
        below = _sample(10, 100, bbox_iou=0.4)
        assert metric.compute([above, below], ["", ""]) == pytest.approx(0.5)

    def test_empty_input(self, metric):
        assert metric.compute([], []) == pytest.approx(0.0)


class TestPassRate:
    @pytest.fixture(params=[0.3, 0.5, 0.7, 0.9])
    def threshold(self, request):
        return request.param

    def test_name_includes_threshold(self, threshold):
        from oellm.contrib.region_reasoner.metrics import PassRate

        pr = PassRate(threshold)
        assert pr.name == f"pass_rate_{threshold}"

    def test_is_base_metric(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        assert isinstance(PassRate(0.5), BaseMetric)

    def test_all_pass(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        s = _sample(100, 100)
        pr = PassRate(0.5)
        assert pr.compute([s, s], ["", ""]) == pytest.approx(1.0)

    def test_none_pass(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        s = _sample(0, 100)
        pr = PassRate(0.3)
        assert pr.compute([s], [""]) == pytest.approx(0.0)

    def test_half_pass(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        perfect = _sample(100, 100)
        zero = _sample(0, 100)
        pr = PassRate(0.5)
        score = pr.compute([perfect, zero], ["", ""])
        assert score == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        with pytest.raises(ValueError):
            PassRate(0.0)
        with pytest.raises(ValueError):
            PassRate(1.0)
        with pytest.raises(ValueError):
            PassRate(-0.1)

    def test_empty_input(self):
        from oellm.contrib.region_reasoner.metrics import PassRate

        assert PassRate(0.5).compute([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Suite plugin protocol
# ---------------------------------------------------------------------------


class TestSuiteProtocol:
    @pytest.fixture
    def suite(self):
        import oellm.contrib.region_reasoner.suite as s

        return s

    def test_suite_name(self, suite):
        assert suite.SUITE_NAME == "region_reasoner"

    def test_cluster_env_vars_declared(self, suite):
        assert "REGION_REASONER_DIR" in suite.CLUSTER_ENV_VARS

    def test_task_groups_structure(self, suite):
        tg = suite.TASK_GROUPS
        assert "task_metrics" in tg
        assert "task_groups" in tg
        assert RR_TASK_NAME in tg["task_metrics"]
        assert RR_TASK_GROUP in tg["task_groups"]

    def test_detect_model_flags_region_reasoner_model(self, suite):
        assert suite.detect_model_flags("lmsdss/RegionReasoner-7B") == "vision_reasoner"

    def test_detect_model_flags_qwen2_model(self, suite):
        assert suite.detect_model_flags("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2"

    def test_detect_model_flags_qwen1_model(self, suite):
        assert suite.detect_model_flags("Qwen/Qwen-VL-Chat") == "qwen"

    def test_detect_model_flags_unknown_defaults_to_vision_reasoner(self, suite):
        assert suite.detect_model_flags("some/unknown-model") == "vision_reasoner"

    def test_parse_results_valid_json(self, suite):
        data = {
            "model_name_or_path": "/path/to/model",
            "results": {
                RR_TASK_NAME: {
                    "gIoU": 0.42,
                    "cIoU": 0.45,
                    "bbox_AP": 0.38,
                }
            },
            "configs": {RR_TASK_NAME: {"num_fewshot": 0}},
        }
        result = suite.parse_results(data)
        assert result is not None
        model_id, task_name, n_shot, metrics = result
        assert model_id == "/path/to/model"
        assert task_name == RR_TASK_NAME
        assert n_shot == 0
        assert metrics["gIoU"] == pytest.approx(0.42)

    def test_parse_results_non_matching_json_returns_none(self, suite):
        # lm-eval output format — should not be parsed by this suite
        data = {
            "model_name": "some_model",
            "results": {"mmlu": {"acc,none": 0.55}},
            "n-shot": {"mmlu": 5},
        }
        assert suite.parse_results(data) is None

    def test_parse_results_empty_results_returns_none(self, suite):
        assert suite.parse_results({}) is None


# ---------------------------------------------------------------------------
# ModelAdapter
# ---------------------------------------------------------------------------


class TestRegionReasonerModelAdapter:
    @pytest.fixture
    def adapter_cls(self):
        from oellm.contrib.region_reasoner.adapter import RegionReasonerModelAdapter
        from oellm.core.base_model_adapter import BaseModelAdapter

        return RegionReasonerModelAdapter, BaseModelAdapter

    def test_is_base_model_adapter(self, adapter_cls):
        cls, base = adapter_cls
        assert issubclass(cls, base)

    def test_contrib_flags_region_reasoner(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("lmsdss/RegionReasoner-7B").to_contrib_flags() == "vision_reasoner"

    def test_contrib_flags_qwen2(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen2.5-VL-7B").to_contrib_flags() == "qwen2"

    def test_contrib_flags_qwen(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("Qwen/Qwen-VL-Chat").to_contrib_flags() == "qwen"

    def test_contrib_flags_unknown_defaults_to_vision_reasoner(self, adapter_cls):
        cls, _ = adapter_cls
        assert cls("some/unknown-model").to_contrib_flags() == "vision_reasoner"

    def test_detect_model_flags_delegates_to_adapter(self):
        import oellm.contrib.region_reasoner.suite as s

        assert s.detect_model_flags("lmsdss/RegionReasoner-7B") == "vision_reasoner"
        assert s.detect_model_flags("Qwen/Qwen2.5-VL-7B") == "qwen2"


# ---------------------------------------------------------------------------
# Schedule evals dry-run integration
# ---------------------------------------------------------------------------


class TestRegionReasonerSchedule:
    def test_schedule_evals_dry_run(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.main._load_cluster_env"),
            patch("oellm.main._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmsdss/RegionReasoner-7B",
                task_groups=RR_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch_files) == 1
        sbatch_content = sbatch_files[0].read_text()
        # The contrib catch-all case must be present
        assert "oellm.contrib.dispatch" in sbatch_content

    def test_jobs_csv_has_region_reasoner_suite(self, tmp_path):
        import pandas as pd

        from oellm.main import schedule_evals

        with (
            patch("oellm.main._load_cluster_env"),
            patch("oellm.main._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmsdss/RegionReasoner-7B",
                task_groups=RR_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        # eval_suite should start with "region_reasoner"
        assert all(s.startswith("region_reasoner") for s in df["eval_suite"])
        assert set(df["task_path"]) == {RR_TASK_NAME}


# ---------------------------------------------------------------------------
# _aggregate_shards (actual metric computation path)
# ---------------------------------------------------------------------------


class TestAggregateShards:
    """Verify _aggregate_shards routes through metrics.py BaseMetric classes."""

    def _write_shard(self, shard_dir, idx, samples):
        path = shard_dir / f"output_{idx}.json"
        path.write_text(json.dumps(samples))

    def test_perfect_overlap(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 100, "union": 100, "bbox_iou": 1.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(1.0)
        assert m["cIoU"] == pytest.approx(1.0)
        assert m["bbox_AP"] == pytest.approx(1.0)
        assert m["pass_rate_0.3"] == pytest.approx(1.0)
        assert m["pass_rate_0.9"] == pytest.approx(1.0)

    def test_zero_overlap(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 0, "union": 200, "bbox_iou": 0.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(0.0)
        assert m["cIoU"] == pytest.approx(0.0)
        assert m["bbox_AP"] == pytest.approx(0.0)
        assert m["pass_rate_0.3"] == pytest.approx(0.0)

    def test_pass_rates_differ_across_thresholds(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [
                {"intersection": 100, "union": 100, "bbox_iou": 1.0},
                {"intersection": 10, "union": 200, "bbox_iou": 0.05},
            ],
        )
        m = _aggregate_shards(str(tmp_path))
        # mask IoU=1.0 passes all; mask IoU=0.05 passes none
        assert m["pass_rate_0.3"] == pytest.approx(0.5)
        assert m["pass_rate_0.5"] == pytest.approx(0.5)
        assert m["pass_rate_0.7"] == pytest.approx(0.5)
        assert m["pass_rate_0.9"] == pytest.approx(0.5)

    def test_pass_rates_actually_differ(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [
                {"intersection": 100, "union": 100, "bbox_iou": 1.0},
                {"intersection": 64, "union": 100, "bbox_iou": 0.64},
            ],
        )
        m = _aggregate_shards(str(tmp_path))
        # mask IoU=1.0 passes all; mask IoU=0.64 passes 0.3 and 0.5 but not 0.7, 0.9
        assert m["pass_rate_0.3"] == pytest.approx(1.0)
        assert m["pass_rate_0.5"] == pytest.approx(1.0)
        assert m["pass_rate_0.7"] == pytest.approx(0.5)
        assert m["pass_rate_0.9"] == pytest.approx(0.5)

    def test_multiple_shards_aggregated(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(
            tmp_path,
            0,
            [{"intersection": 100, "union": 100, "bbox_iou": 1.0}],
        )
        self._write_shard(
            tmp_path,
            1,
            [{"intersection": 0, "union": 100, "bbox_iou": 0.0}],
        )
        m = _aggregate_shards(str(tmp_path))
        assert m["gIoU"] == pytest.approx(0.5)
        assert m["cIoU"] == pytest.approx(0.5)

    def test_no_shard_files_raises(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        with pytest.raises(RuntimeError, match="No shard output files"):
            _aggregate_shards(str(tmp_path))

    def test_empty_shard_raises(self, tmp_path):
        from oellm.contrib.region_reasoner.suite import _aggregate_shards

        self._write_shard(tmp_path, 0, [])
        with pytest.raises(RuntimeError, match="No samples found"):
            _aggregate_shards(str(tmp_path))


# ---------------------------------------------------------------------------
# collect_results compatibility
# ---------------------------------------------------------------------------


class TestCollectResultsCompatibility:
    """Verify collect_results() parses RegionReasoner output without modification."""

    def test_collect_results_parses_region_reasoner_json(self, tmp_path):
        import pandas as pd

        from oellm.main import collect_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write a mock RegionReasoner output JSON (lmms-eval-compatible format)
        mock_output = {
            "model_name_or_path": "/cluster/models/RegionReasoner-7B",
            "results": {
                RR_TASK_NAME: {
                    "gIoU": 0.42,
                    "cIoU": 0.45,
                    "bbox_AP": 0.38,
                    "pass_rate_0.3": 0.71,
                    "pass_rate_0.5": 0.55,
                }
            },
            "configs": {RR_TASK_NAME: {"num_fewshot": 0}},
        }
        (results_dir / "abc123.json").write_text(json.dumps(mock_output))

        output_csv = str(tmp_path / "results.csv")
        collect_results(str(tmp_path), output_csv=output_csv)

        assert Path(output_csv).exists()
        df = pd.read_csv(output_csv)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["task"] == RR_TASK_NAME
        assert row["metric_name"] in ("gIoU", "gIoU,none")  # primary metric
        assert float(row["performance"]) == pytest.approx(0.42)
        assert row["model_name"] == "/cluster/models/RegionReasoner-7B"
