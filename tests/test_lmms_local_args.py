"""lmms-eval device placement is decided by the job script, not baked at schedule time."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from oellm.core.base_model_adapter import DefaultHFAdapter, VLLMAdapter
from oellm.main import schedule_evals


def test_lmms_args_leave_device_to_the_job_script():
    args = DefaultHFAdapter().to_lmms_eval_args()
    assert args.startswith("pretrained=$model_path")
    assert "device_map" not in args
    assert "$_lmms_extra_args" in args

    vllm_args = VLLMAdapter(
        gpu_memory_utilization=0.93, max_num_seqs=96, max_model_len=65536
    ).to_lmms_eval_args()
    assert vllm_args.startswith("model=$model_path")
    assert "device_map" not in vllm_args
    assert "$_lmms_extra_args" not in vllm_args
    assert "tensor_parallel_size=1" in vllm_args
    assert "gpu_memory_utilization=0.93" in vllm_args
    assert "max_num_seqs=96" in vllm_args
    assert "max_model_len=65536" in vllm_args


def _render(tmp_path: Path) -> Path:
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
        patch("oellm.runner.detect_lmms_model_type", return_value="llava_hf"),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="llava-hf/llava-interleave-qwen-0.5b-hf",
            task_groups="image-realworldqa",
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )
    return next(tmp_path.glob("**/submit_evals.sbatch"))


def test_job_script_resolves_lmms_device_at_run_time(tmp_path):
    script_path = _render(tmp_path)
    script = script_path.read_text()
    assert 'case ",${LMMS_MODEL_ARGS:-}," in' in script
    assert '_lmms_extra_args=",device=mps,device_map=mps"' in script
    assert '_lmms_extra_args=",device_map=auto"' in script
    assert '--model_args "pretrained=$model_path$_lmms_extra_args"' in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_lmms_model_args_override_is_recorded_in_provenance(tmp_path):
    with patch.dict(os.environ, {"LMMS_MODEL_ARGS": "device=cpu,device_map=cpu"}):
        script_path = _render(tmp_path)
    provenance = json.loads((script_path.parent / "provenance.json").read_text())
    assert provenance["lmms_model_args"] == "device=cpu,device_map=cpu"
    # The value is read on the node, never substituted into the script.
    assert "${LMMS_MODEL_ARGS:-}" in script_path.read_text()
