"""Regenerate the DocVQA 2026 scorer parity fixture from the official code.

Development tool, not part of any run. It imports the upstream
``eval_utils.py`` from a local clone of https://github.com/VLR-CVC/DocVQA2026
and records what it returns for a grid of (prediction, ground truth) pairs.
tests/test_docvqa2026_parity.py then holds our vendored copy to that record,
so the test suite never needs the upstream clone.

    python scripts/gen_docvqa2026_parity_fixture.py --ref ~/Projects/DocVQA2026

Ground truths come from the real val split, read via the parquet text columns
only, so no image bytes are downloaded.
"""

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "docvqa2026_parity.json"


def load_reference(ref_dir: Path):
    path = ref_dir / "eval_utils.py"
    if not path.exists():
        raise SystemExit(f"upstream eval_utils.py not found at {path}")
    spec = importlib.util.spec_from_file_location("docvqa2026_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_ground_truths() -> list[str]:
    """Every answer in the val split, read without touching the images."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open("datasets/VLR-CVC/DocVQA-2026/val.parquet", "rb") as f:
        table = pq.ParquetFile(f).read(columns=["answers"])
    out: list[str] = []
    for row in table.to_pylist():
        out.extend(row["answers"]["answer"])
    return out


def predictions_for(gt: str) -> list[str]:
    """Predictions probing each branch of the scorer for one ground truth."""
    body = gt.strip()
    return [
        f"FINAL ANSWER: {body}",  # exact
        body,  # marker missing -> always wrong
        f"final answer: {body}",  # marker is case-sensitive
        f"reasoning...\nFINAL ANSWER: {body}",
        f"FINAL ANSWER: {body}   ",  # trailing space
        f"FINAL ANSWER:{body}",  # no space after marker
        f"FINAL ANSWER: The answer is {body}",
        f"FINAL ANSWER: {body}.",  # trailing punctuation
        f"FINAL ANSWER: the {body}",  # article, stripped by clean_text
        f"FINAL ANSWER: {body.upper()}",
        f"FINAL ANSWER: {body} kg",  # spurious unit
        f"FINAL ANSWER: {body}%",  # percent attached
        "FINAL ANSWER: Unknown",
        "FINAL ANSWER: ",  # empty answer
        f"FINAL ANSWER: {body}\nFINAL ANSWER: {body}",  # split()[-1] wins
    ]


# Pairs that do not depend on the dataset: numeric, unit, date and version
# handling, and ANLS either side of the 0.9 threshold.
SYNTHETIC: list[tuple[str, str]] = [
    ("FINAL ANSWER: 4", "4"),
    ("FINAL ANSWER: 4.0", "4"),
    ("FINAL ANSWER: 4", "4.0"),
    ("FINAL ANSWER: 04", "4"),
    ("FINAL ANSWER: four", "4"),
    ("FINAL ANSWER: 1000", "1,000"),
    ("FINAL ANSWER: 1,000", "1000"),
    ("FINAL ANSWER: 50 kg", "50 kg"),
    ("FINAL ANSWER: 50kg", "50 kg"),
    ("FINAL ANSWER: 50 kilograms", "50 kg"),
    ("FINAL ANSWER: 50 g", "50 kg"),
    ("FINAL ANSWER: 50", "50 kg"),
    ("FINAL ANSWER: 50%", "50%"),
    ("FINAL ANSWER: 50 %", "50%"),
    ("FINAL ANSWER: -3.5 m", "-3.5 m"),
    ("FINAL ANSWER: 2024-01-01", "2024-01-01"),
    ("FINAL ANSWER: 2024-01-02", "2024-01-01"),
    ("FINAL ANSWER: Jan 1st 24", "2024-01-01"),
    ("FINAL ANSWER: January 1, 2024", "2024-01-01"),
    ("FINAL ANSWER: 01/01/2024", "2024-01-01"),
    ("FINAL ANSWER: 12.0.0", "12.0.0"),
    ("FINAL ANSWER: 12.0.1", "12.0.0"),
    ("FINAL ANSWER: 1.2.3", "1.2.3"),
    ("FINAL ANSWER: green", "['olive green', 'green', 'dark green']"),
    ("FINAL ANSWER: olive green", "['olive green', 'green', 'dark green']"),
    ("FINAL ANSWER: blue", "['olive green', 'green', 'dark green']"),
    ("FINAL ANSWER: ['green']", "['olive green', 'green', 'dark green']"),
    ("FINAL ANSWER: Answer A, Answer B", "Answer A, Answer B"),
    ("FINAL ANSWER: Answer A and Answer B", "Answer A, Answer B"),
    ("FINAL ANSWER: the wrench", "wrench"),
    ("FINAL ANSWER: a wrench!", "wrench"),
    ("FINAL ANSWER: wrenchh", "wrench"),
    ("FINAL ANSWER: wrenchhh", "wrench"),
    ("FINAL ANSWER: U.S. Marines", "U.S. Marines"),
    ("FINAL ANSWER: US Marines", "U.S. Marines"),
    ("FINAL ANSWER: Macadam and Gravel", "Macadam & Gravel"),
    ("FINAL ANSWER: Macadam & Gravel", "Macadam & Gravel"),
    ("no marker at all", "4"),
    ("", "4"),
    ("FINAL ANSWER: 4", ""),
    ("FINAL ANSWER: ", ""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="clone of VLR-CVC/DocVQA2026")
    args = ap.parse_args()

    ref = load_reference(args.ref)
    cases: list[tuple[str, str]] = list(SYNTHETIC)
    for gt in real_ground_truths():
        cases.extend((pred, gt) for pred in predictions_for(gt))

    records = []
    for pred, gt in cases:
        correct, extracted = ref.evaluate_docvqa_prediction(pred, gt)
        records.append(
            {
                "prediction": pred,
                "ground_truth": gt,
                "correct": bool(correct),
                "extracted": extracted,
            }
        )

    payload = {
        "source": "https://github.com/VLR-CVC/DocVQA2026 eval_utils.py",
        "prompt": ref.get_evaluation_prompt(),
        "cases": records,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    n_true = sum(r["correct"] for r in records)
    print(
        f"wrote {FIXTURE}: {len(records)} cases ({n_true} correct, {len(records) - n_true} incorrect)"
    )


if __name__ == "__main__":
    main()
