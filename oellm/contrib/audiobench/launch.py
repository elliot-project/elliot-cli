"""Run one AudioBench evaluation, optionally on your own checkpoint.

AudioBench's loaders read their weights location from a module variable
(e.g. ``model_path = "Qwen/Qwen2-Audio-7B-Instruct"`` in
``model_src/qwen2_audio_7b_instruct.py``). This script points that variable at
the checkpoint before AudioBench loads the model, and writes AudioBench's
predictions and score file into ``--log-dir`` instead of the shared
``log_for_all_models`` folder of the clone.

Run by :func:`oellm.contrib.audiobench.suite.run` in a subprocess, with the
AudioBench clone as working directory (some loaders use paths relative to it).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audiobench-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model-name", required=True, help="AudioBench dispatch key")
    p.add_argument("--metrics", required=True)
    p.add_argument("--number-of-samples", type=int, default=-1)
    p.add_argument("--module", help="model_src module whose weights to replace")
    p.add_argument("--variable", help="module variable holding the weights location")
    p.add_argument("--checkpoint", help="checkpoint to load instead")
    args = p.parse_args(argv)

    src = Path(args.audiobench_dir).resolve() / "src"
    # AudioBench imports its own modules by bare name (``from dataset import
    # Dataset``); keep this script's folder off the path so nothing shadows them.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(src)] + [entry for entry in sys.path if entry != here]
    os.chdir(args.audiobench_dir)

    import main_evaluate

    main_evaluate.file_save_folder = str(Path(args.log_dir).resolve())

    if args.checkpoint:
        module = importlib.import_module(f"model_src.{args.module}")
        stock = getattr(module, args.variable)
        setattr(module, args.variable, args.checkpoint)
        print(
            f"AudioBench {args.model_name}: loading {args.checkpoint} instead of {stock}"
        )

    main_evaluate.main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        metrics=args.metrics,
        overwrite=True,
        number_of_samples=args.number_of_samples,
    )


if __name__ == "__main__":
    main()
