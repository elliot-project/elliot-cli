"""The vendored DocVQA 2026 scorer must agree with the official one exactly.

tests/fixtures/docvqa2026_parity.json records what the upstream
``eval_utils.py`` returns for 1241 (prediction, ground truth) pairs — every
answer in the val split crossed with predictions that probe each branch, plus
synthetic numeric/unit/date/version/ANLS cases. Regenerate it with
scripts/gen_docvqa2026_parity_fixture.py against a fresh clone.

The fixture is what makes this test self-contained: parity is verified on
every CI run without cloning the competition repository.
"""

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "docvqa2026_parity.json"
REFERENCE_CLONE = Path.home() / "Projects" / "DocVQA2026"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def scorer():
    from oellm.contrib.docvqa2026 import _vendor_eval_utils

    return _vendor_eval_utils


class TestScorerMatchesOfficial:
    def test_every_case_agrees(self, golden, scorer):
        mismatches = []
        for case in golden["cases"]:
            correct, extracted = scorer.evaluate_docvqa_prediction(
                case["prediction"], case["ground_truth"]
            )
            if bool(correct) != case["correct"] or extracted != case["extracted"]:
                mismatches.append(
                    f"  pred={case['prediction']!r} gt={case['ground_truth']!r}: "
                    f"official=({case['correct']}, {case['extracted']!r}) "
                    f"ours=({bool(correct)}, {extracted!r})"
                )
        assert not mismatches, (
            "scorer diverged from the official implementation:\n"
            + "\n".join(mismatches[:20])
        )

    def test_prompt_is_byte_identical(self, golden, scorer):
        assert scorer.get_evaluation_prompt() == golden["prompt"]

    def test_fixture_discriminates(self, golden):
        """A fixture of all-True (or all-False) cases would pass vacuously."""
        verdicts = [c["correct"] for c in golden["cases"]]
        assert len(verdicts) > 1000
        assert 0.2 < sum(verdicts) / len(verdicts) < 0.8


class TestDocumentedQuirksHold:
    """Behaviour the competition scorer depends on, stated explicitly.

    These are the branches most likely to be "cleaned up" by a later edit;
    each one changes published scores if it moves.
    """

    def test_missing_marker_is_always_wrong(self, scorer):
        correct, extracted = scorer.evaluate_docvqa_prediction("4", "4")
        assert correct is False
        assert extracted == "4"

    def test_numeric_ground_truth_skips_the_anls_fallback(self, scorer):
        # "four" would clear the 0.9 ANLS bar against no candidate, but a
        # numeric ground truth short-circuits before ANLS is ever consulted.
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: four", "4")[0] is False

    def test_value_equal_unit_different_is_wrong(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: 50 g", "50 kg")[0] is False
        )

    def test_textual_date_matches_iso_ground_truth(self, scorer):
        assert (
            scorer.evaluate_docvqa_prediction("FINAL ANSWER: Jan 1st 24", "2024-01-01")[0]
            is True
        )

    def test_list_ground_truth_accepts_any_candidate(self, scorer):
        gt = "['olive green', 'green', 'dark green']"
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: green", gt)[0] is True
        assert scorer.evaluate_docvqa_prediction("FINAL ANSWER: blue", gt)[0] is False

    def test_last_marker_wins(self, scorer):
        correct, extracted = scorer.evaluate_docvqa_prediction(
            "FINAL ANSWER: wrong\nFINAL ANSWER: 4", "4"
        )
        assert extracted == "4"
        assert correct is True


@pytest.mark.skipif(
    not (REFERENCE_CLONE / "eval_utils.py").exists(),
    reason="upstream DocVQA2026 clone not present",
)
def test_fixture_still_matches_upstream():
    """Catches upstream edits: run where a clone exists, skipped in CI."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "docvqa2026_upstream", REFERENCE_CLONE / "eval_utils.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    golden = json.loads(FIXTURE.read_text())
    assert upstream.get_evaluation_prompt() == golden["prompt"]
    for case in golden["cases"]:
        correct, extracted = upstream.evaluate_docvqa_prediction(
            case["prediction"], case["ground_truth"]
        )
        assert (bool(correct), extracted) == (case["correct"], case["extracted"]), (
            f"upstream changed for pred={case['prediction']!r} gt={case['ground_truth']!r}"
        )
