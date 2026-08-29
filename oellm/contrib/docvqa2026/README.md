# DocVQA 2026

Reasoning questions over multi-page documents in eight domains, from the
ICDAR 2026 competition ([site](https://www.docvqa.org/challenges/2026),
[eval code](https://github.com/VLR-CVC/DocVQA2026),
[dataset](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026)).

```bash
oellm-eval schedule \
  --models HuggingFaceTB/SmolVLM-256M-Instruct \
  --task-groups image-docvqa2026 \
  --venv-path /path/to/.venv
```

**Metrics:** `accuracy` (primary, over all questions), `macro_accuracy` (mean
of the eight per-domain rates), and `acc_<domain>` for each domain.

Both averages are reported because they answer different questions. On `val`
they are equal — each of the eight domains holds exactly 10 of the 80
questions — so the competition's per-domain table reproduces either way. They
would diverge on a split with uneven domains.

## Only `val` is scorable

`val` is 25 documents and 80 questions. The `test` split ships with its
answers withheld and is graded solely on the
[RRC platform](https://rrc.cvc.uab.es/?ch=34).

## Scoring is the competition's own code

`_vendor_eval_utils.py` is copied byte-for-byte from the official
`eval_utils.py` and excluded from ruff, so no formatter can drift it. Its
behaviour is pinned by `tests/test_docvqa2026_parity.py` against a fixture of
1241 recorded upstream verdicts, which is why the test suite never needs a
clone of the competition repository.

Three behaviours of that scorer decide most scores, and none of them are
bugs to fix here:

- A prediction without the `FINAL ANSWER:` marker is wrong whatever it says,
  so the prompt in `prompts.py` is the competition's, verbatim.
- When the ground truth parses as a number, a failed strict match returns
  wrong **without** the ANLS fallback. 37 of the 80 val answers parse as
  numbers, so nearly half the benchmark is exact number-and-unit equality.
- Values must match *and* units must match; `50 g` scores zero against
  `50 kg`.

## Pages, and why your number may not be comparable

Each question is asked against its whole document — 905 page images across 25
documents, about 36 pages each, roughly 50k image tokens per question. Models
that cannot hold that need `DOCVQA2026_MAX_PAGES`:

```bash
DOCVQA2026_MAX_PAGES=4 oellm-eval schedule --task-groups image-docvqa2026 ...
```

A capped run records `max_pages` and `n_truncated_documents` in its results
and logs a warning. **Capped scores are not comparable with the competition
leaderboard**, which grades the full document. The published baselines
(Gemini 3 Pro 0.375, GPT-5.2 0.350) are frontier API models reading every
page; a small local VLM on a handful of pages is measuring something else.

Any checkpoint whose processor exposes a chat template accepting interleaved
images works — SmolVLM, Idefics, Qwen-VL — so there is no per-family branching
in the adapter.
