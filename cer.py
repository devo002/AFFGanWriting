#!/usr/bin/env python3
import sys
from pathlib import Path

def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0]*n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev = curr
    return prev[n]

def parse_gt_pred(fname: str):
    stem = Path(fname).stem
    after_dot = stem.split('.', 1)[1] if '.' in stem else stem
    gt, pred = after_dot.split('-', 1)
    return gt, pred

def main(folder):
    paths = list(Path(folder).glob("*.png"))
    if not paths:
        print("No .png files found.")
        return
    total_edits = 0
    total_chars = 0
    for p in paths:
        gt, pred = parse_gt_pred(p.name)
        total_edits += levenshtein(gt, pred)
        total_chars += len(gt)

    avg_cer = total_edits / total_chars if total_chars else float("nan")
    print(f"Total character errors: {total_edits}")
    print(f"Average CER: {avg_cer:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cer_report.py <folder_path>")
        sys.exit(1)
    main(sys.argv[1])
