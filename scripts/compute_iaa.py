"""Compute inter-annotator agreement (IAA) on a subset of the test set.

This script assumes:
- data/test/questions.txt exists
- data/test/iaa_question_line_numbers.txt lists 1-based line numbers for the IAA subset
- Two annotators provide answers for the same subset lines in two files, one answer per line:
    data/test/iaa_annotator1.txt
    data/test/iaa_annotator2.txt

We report:
- exact match (EM) after normalization
- token-level F1 after normalization (SQuAD-style)
"""

from __future__ import annotations

import argparse
import collections
import re
import string
from pathlib import Path


def normalize_answer(text: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def rm_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return " ".join(remove_articles(rm_punc(text.lower())).split())


def get_tokens(text: str) -> list[str]:
    t = normalize_answer(text)
    return t.split() if t else []


def f1(pred: str, truth: str) -> float:
    pt, tt = get_tokens(pred), get_tokens(truth)
    if not pt or not tt:
        return float(pt == tt)
    common = collections.Counter(pt) & collections.Counter(tt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(tt)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--a1",
        type=Path,
        default=Path("data/test/iaa_annotator1.txt"),
        help="Annotator 1 answers for IAA subset (one line per subset item).",
    )
    ap.add_argument(
        "--a2",
        type=Path,
        default=Path("data/test/iaa_annotator2.txt"),
        help="Annotator 2 answers for IAA subset (one line per subset item).",
    )
    ap.add_argument(
        "--subset",
        type=Path,
        default=Path("data/test/iaa_question_line_numbers.txt"),
        help="1-based line numbers from data/test/questions.txt used for IAA.",
    )
    args = ap.parse_args()

    subset_lines = [int(x.strip()) for x in args.subset.read_text(encoding="utf-8").splitlines() if x.strip()]
    a1 = args.a1.read_text(encoding="utf-8").splitlines()
    a2 = args.a2.read_text(encoding="utf-8").splitlines()
    if len(a1) != len(subset_lines) or len(a2) != len(subset_lines):
        raise SystemExit(
            f"IAA files must have exactly {len(subset_lines)} lines (one per subset item). "
            f"Got a1={len(a1)} a2={len(a2)}."
        )

    ems = []
    f1s = []
    for x, y in zip(a1, a2):
        ems.append(int(normalize_answer(x) == normalize_answer(y)))
        f1s.append(f1(x, y))

    n = len(ems)
    print(f"Subset size: {n}")
    print(f"IAA Exact Match: {100.0 * sum(ems) / n:.2f}")
    print(f"IAA Token F1:    {100.0 * sum(f1s) / n:.2f}")


if __name__ == "__main__":
    main()

