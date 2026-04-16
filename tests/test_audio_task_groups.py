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

AUDIO_TASK_GROUP = "audio-understanding"

# Tasks included in the curated audio-understanding suite. This is a subset of
# the full lmms-eval audio coverage — individual audio-* task groups cover the
# rest (see test_individual_audio_groups_present below).
EXPECTED_TASKS = {
    "librispeech_test_clean",
    "fleurs_en",
    "covost2_en_zh_test",
    "clotho_aqa_test",
    "vocalsound_test",
    "muchomusic",
    "air_bench_chat",
    "voicebench_commoneval",
}

EXPECTED_DATASETS = {
    "lmms-lab/librispeech",
    "lmms-lab/fleurs",
    "lmms-lab/covost2_en-zh",
    "lmms-lab/ClothoAQA",
    "lmms-lab/vocalsound",
    "lmms-lab/muchomusic",
    "lmms-lab/AIR_Bench",
    "lmms-lab/voicebench",
}

# All individual audio-* groups that must be registered so users can target
# a single benchmark from the CLI.
INDIVIDUAL_AUDIO_GROUPS = [
    "audio-librispeech",
    "audio-librispeech-all",
    "audio-common-voice-15",
    "audio-gigaspeech",
    "audio-tedlium",
    "audio-people-speech",
    "audio-voxpopuli",
    "audio-ami",
    "audio-wenet-speech",
    "audio-covost2",
    "audio-fleurs",
    "audio-alpaca-audio",
    "audio-clotho-aqa",
    "audio-openhermes",
    "audio-wavcaps",
    "audio-air-bench-chat",
    "audio-air-bench-foundation",
    "audio-muchomusic",
    "audio-vocalsound",
    "audio-step2-paralinguistic",
    "audio-cn-college-listen-mcq",
    "audio-dream-tts-mcq",
    "audio-voicebench",
]


class TestAudioTaskGroupInRegistry:
    def test_audio_understanding_present_in_yaml(self):
        all_groups = get_all_task_group_names()
        assert AUDIO_TASK_GROUP in all_groups

    def test_audio_understanding_suite_is_lmms_eval(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        suite = data["task_groups"][AUDIO_TASK_GROUP]["suite"]
        assert suite == "lmms_eval"

    def test_audio_understanding_has_eight_tasks(self):
        data = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
        tasks = data["task_groups"][AUDIO_TASK_GROUP]["tasks"]
        assert len(tasks) == 8

    def test_individual_audio_groups_present(self):
        all_groups = get_all_task_group_names()
        for name in INDIVIDUAL_AUDIO_GROUPS:
            assert name in all_groups, f"{name} not in task group registry"


class TestAudioTaskGroupExpansion:
    def test_expands_to_correct_task_names(self):
        results = _expand_task_groups([AUDIO_TASK_GROUP])
        task_names = {r.task for r in results}
        assert task_names == EXPECTED_TASKS

    def test_all_tasks_have_zero_shot(self):
        results = _expand_task_groups([AUDIO_TASK_GROUP])
        for r in results:
            assert r.n_shot == 0, f"{r.task} has n_shot={r.n_shot}, expected 0"

    def test_all_tasks_route_to_lmms_eval(self):
        results = _expand_task_groups([AUDIO_TASK_GROUP])
        for r in results:
            assert r.suite == "lmms_eval", (
                f"{r.task} has suite='{r.suite}', expected 'lmms_eval'"
            )

    def test_expand_individual_audio_group(self):
        results = _expand_task_groups(["audio-librispeech"])
        assert len(results) == 1
        assert results[0].task == "librispeech_test_clean"
        assert results[0].suite == "lmms_eval"

    def test_expand_librispeech_all_has_four_splits(self):
        results = _expand_task_groups(["audio-librispeech-all"])
        task_names = {r.task for r in results}
        assert task_names == {
            "librispeech_dev_clean",
            "librispeech_dev_other",
            "librispeech_test_clean",
            "librispeech_test_other",
        }

    def test_expand_air_bench_chat_has_four_subsets(self):
        results = _expand_task_groups(["audio-air-bench-chat"])
        task_names = {r.task for r in results}
        assert task_names == {
            "air_bench_chat_sound",
            "air_bench_chat_music",
            "air_bench_chat_speech",
            "air_bench_chat_mixed",
        }


class TestAudioTaskGroupDatasetSpecs:
    def test_all_expected_datasets_present(self):
        specs = _collect_dataset_specs([AUDIO_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert repo_ids == EXPECTED_DATASETS

    def test_no_duplicate_dataset_specs(self):
        specs = _collect_dataset_specs([AUDIO_TASK_GROUP])
        keys = [(s.repo_id, s.subset) for s in specs]
        assert len(keys) == len(set(keys)), "Duplicate dataset specs found"

    def test_needs_snapshot_download_flag_set_on_specs(self):
        """audio-* groups must mark their DatasetSpecs as
        needs_snapshot_download=True so _pre_download_datasets_from_specs
        mirrors the whole repo (load_dataset alone doesn't materialize the
        loose .flac / .wav blobs that lmms-eval reads at runtime)."""
        specs = _collect_dataset_specs([AUDIO_TASK_GROUP])
        assert specs, "No dataset specs returned"
        for s in specs:
            assert s.needs_snapshot_download, (
                f"DatasetSpec for {s.repo_id} missing needs_snapshot_download=True flag"
            )

    def test_librispeech_dataset_included(self):
        specs = _collect_dataset_specs([AUDIO_TASK_GROUP])
        repo_ids = {s.repo_id for s in specs}
        assert "lmms-lab/librispeech" in repo_ids


class TestAudioTaskGroupScheduleEvals:
    """Verify audio-understanding integrates with the schedule_evals dry-run path."""

    def test_schedule_evals_dry_run_audio(self, tmp_path):
        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch(
                "oellm.runner.detect_lmms_model_type",
                return_value="qwen2_audio",
            ),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="Qwen/Qwen2-Audio-7B-Instruct",
                task_groups=AUDIO_TASK_GROUP,
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
                return_value="qwen2_audio",
            ),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="Qwen/Qwen2-Audio-7B-Instruct",
                task_groups=AUDIO_TASK_GROUP,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

        csv_files = list(tmp_path.glob("**/jobs.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        assert all(s.startswith("lmms_eval") for s in df["eval_suite"].unique())
        assert set(df["task_path"].unique()) == EXPECTED_TASKS


class TestAudioModelAdapters:
    """Verify audio-specific model adapter detection."""

    def test_qwen2_audio_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("Qwen/Qwen2-Audio-7B-Instruct") == "qwen2_audio"

    def test_qwen2_5_audio_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("Qwen/Qwen2.5-Audio-7B") == "qwen2_5_audio"

    def test_salmonn_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("tsinghua-ee/SALMONN-7B") == "salmonn"

    def test_audio_flamingo_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("nvidia/audio-flamingo-2") == "audio_flamingo"

    def test_ultravox_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("fixie-ai/ultravox-v0_4") == "ultravox"

    def test_phi4_multimodal_detected(self):
        from oellm.constants import detect_lmms_model_type

        assert (
            detect_lmms_model_type("microsoft/Phi-4-multimodal-instruct")
            == "phi4_multimodal"
        )

    def test_qwen2_audio_does_not_route_to_qwen_vl(self):
        """qwen2-audio must resolve to the audio adapter, not the generic qwen_vl
        catch-all. This guards against ordering regressions in LMMS_MODEL_ADAPTERS."""
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("Qwen/Qwen2-Audio-7B") == "qwen2_audio"

    def test_vision_adapters_still_work(self):
        """Adding audio patterns must not regress existing vision adapter detection."""
        from oellm.constants import detect_lmms_model_type

        assert detect_lmms_model_type("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2_5_vl"
        assert detect_lmms_model_type("llava-hf/llava-1.5-7b-hf") == "llava_hf"
        assert detect_lmms_model_type("lmms-lab/llava-onevision-7b") == "llava_onevision"
