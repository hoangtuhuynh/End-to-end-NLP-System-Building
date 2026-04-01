"""Check that questions.txt and reference_answers.txt have the same line count."""
from pathlib import Path


def check_pair(q_path: Path, a_path: Path) -> None:
    q_lines = q_path.read_text(encoding="utf-8").splitlines()
    a_lines = a_path.read_text(encoding="utf-8").splitlines()
    nq, na = len(q_lines), len(a_lines)
    if nq != na:
        raise SystemExit(f"Mismatch {q_path}: {nq} questions vs {na} references")
    print(f"OK {q_path.parent.name}: {nq} pairs")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    check_pair(root / "data" / "test" / "questions.txt", root / "data" / "test" / "reference_answers.txt")
    check_pair(root / "data" / "train" / "questions.txt", root / "data" / "train" / "reference_answers.txt")
