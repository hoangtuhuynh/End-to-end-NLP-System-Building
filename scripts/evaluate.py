"""SQuAD-style EM / F1 versus reference_answers.txt (semicolon-separated alternates)."""
from __future__ import annotations

import argparse
import collections
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
        return int(pt == tt)
    common = collections.Counter(pt) & collections.Counter(tt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(tt)
    return 2 * precision * recall / (precision + recall)


def answer_recall(pred: str, truth: str) -> float:
    """Token recall: what fraction of reference tokens appear in the prediction."""
    pt, tt = get_tokens(pred), get_tokens(truth)
    if not tt:
        return 1.0
    if not pt:
        return 0.0
    cp, ct = collections.Counter(pt), collections.Counter(tt)
    covered = sum((cp & ct).values())
    return covered / sum(ct.values())


def max_over_refs(metric, pred: str, refs: list[str]) -> float:
    return max(metric(pred, r) for r in refs) if refs else 0.0


def parse_refs(line: str) -> list[str]:
    parts = [p.strip() for p in line.split(";")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    args = ap.parse_args()

    preds = args.predictions.read_text(encoding="utf-8").splitlines()
    refs = args.references.read_text(encoding="utf-8").splitlines()
    if len(preds) != len(refs):
        raise SystemExit(f"Length mismatch: {len(preds)} preds vs {len(refs)} refs")

    ems, f1s, ars = [], [], []
    for p, rline in zip(preds, refs):
        rlist = parse_refs(rline)
        ems.append(max(int(normalize_answer(p) == normalize_answer(r)) for r in rlist))
        f1s.append(max_over_refs(f1, p, rlist))
        ars.append(max_over_refs(answer_recall, p, rlist))

    n = len(ems)
    print(f"Examples: {n}")
    print(f"Exact Match:  {100.0 * sum(ems) / n:.2f}")
    print(f"Token F1:     {100.0 * sum(f1s) / n:.2f}")
    print(f"Answer Recall:{100.0 * sum(ars) / n:.2f}")


if __name__ == "__main__":
    main()
