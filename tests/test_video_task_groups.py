import os
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import yaml

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    get_all_task_group_names,
)

VIDEO_TASK_GROUP = "video-understanding"

EXPECTED_TASKS = {
    "mvbench",
    "egoschema",
    "videomme",
    "activitynetqa",
    "longvideobench_val_v",
}

EXPECTED_DATASETS = {
    "OpenGVLab/MVBench",
    "lmms-lab/egoschema",
    "lmms-lab/Video-MME",
    "lmms-lab/ActivityNetQA",
    "longvideobench/LongVideoBench",
}


class TestVideoTaskGroupInRegistry:
    def test_video_understanding_present_in_yaml(self):
        all_groups = get_all_task_group_names()
        assert VIDEO_TASK_GROUP in all_groups

    def test_video_understanding_suite_is_lmms_eval(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        suite = data["task_groups"][VIDEO_TASK_GROUP]["suite"]
        assert suite == "lmms_eval"

    def test_video_understanding_has_five_tasks(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        tasks = data["task_groups"][VIDEO_TASK_GROUP]["tasks"]
        assert len(tasks) == 5

    def test_individual_video_groups_present(self):
        all_groups = get_all_task_group_names()
        for name in [
            "video-mvbench",
            "video-egoschema",
            "video-videomme",
            "video-activitynet-qa",
            "video-longvideobench",
        ]:
            assert name in all_groups, f"{name} not in task group registry"


class TestVideoTaskGroupExpansion:
    def test_expands_to_correct_task_names(self):
        results = _expand_task_groups([VIDEO_TASK_GROUP])
        task_names = {r.task for r in results}
        assert task_names == EXPECTED_TASKS

    def test_all_tasks_have_zero_shot(self):
        results = _expand_task_groups([VIDEO_TASK_GROUP])
        for r in results:
            assert r.n_shot == 0, f"{r.task} has n_shot={r.n_shot}, expected 0"

    def test_all_tasks_route_to_lmms_eval(self):
        results = _expand_task_groups([VIDEO_TASK_GROUP])
        for r in results:
            assert r.suite == "lmms_eval", (
                f"{r.task} has suite='{r.suite}', expected 'lmms_eval'"
            )

    def test_expand_individual_video_group(self):
        results = _expand_task_groups(["video-mvbench"])
        assert len(results) == 1
        assert results[0].task == "mvbench"
        assert results[0].suite == "lmms_eval"


class TestVideoTaskGroupDatasetSpecs:
    def test_all_expected_datasets_present(self):
        specs = _collect_dataset_specs([VIDEO_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert repo_ids == EXPECTED_DATASETS

    def test_no_duplicate_dataset_specs(self):
        specs = _collect_dataset_specs([VIDEO_TASK_GROUP])
        keys = [(s.repo_id, s.subset) for s in specs]
        assert len(keys) == len(set(keys)), "Duplicate dataset specs found"

    def test_videomme_dataset_included(self):
        specs = _collect_dataset_specs([VIDEO_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert "lmms-lab/Video-MME" in repo_ids


class TestVideoTaskGroupScheduleEvals:
    """Verify video-understanding integrates with the schedule_evals dry-run path."""

    def test_schedule_evals_dry_run_video(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch(
                "oellm.runner.detect_lmms_model_type",
                return_value="llava_onevision",
            ),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmms-lab/llava-onevision-7b",
                task_groups=VIDEO_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch_files) == 1
        sbatch_content = sbatch_files[0].read_text()
        assert "lmms_eval" in sbatch_content

    def test_schedule_evals_jobs_csv_has_lmms_eval_suite(self, tmp_path):
        import pandas as pd

        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch(
                "oellm.runner.detect_lmms_model_type",
                return_value="llava_onevision",
            ),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="lmms-lab/llava-onevision-7b",
                task_groups=VIDEO_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        assert all(s.startswith("lmms_eval") for s in df["eval_suite"].unique())
        assert set(df["task_path"].unique()) == EXPECTED_TASKS


class TestVideoModelAdapters:
    """Verify video-specific model adapter detection."""

    def test_llava_onevision_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("lmms-lab/llava-onevision-7b") == "llava_onevision"

    def test_llava_vid_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("llava-vid-7b") == "llava_vid"

    def test_video_llava_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("video-llava-7b") == "video_llava"

    def test_longva_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("longva-7b") == "longva"

    def test_internvideo_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("internvideo2-chat") == "internvideo2"

    def test_generic_llava_still_works(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("llava-hf/llava-1.5-7b-hf") == "llava_hf"

    def test_qwen25_vl_still_works(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2_5_vl"
