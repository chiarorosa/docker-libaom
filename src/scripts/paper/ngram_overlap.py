"""Near-copy check: word n-grams a paper shares with its mold paper.

The ISCAS text must clone the LASCAS lexicon but not its sentences
(ISCAS/CLAUDE.md, sections 2 and 3b). Lists every shared n-gram (default
n=6), so near-verbatim passages are rewritten. Canonical group phrases of
CLAUDE.md B.12 may legitimately appear and are reviewed by hand.

    py src/scripts/paper/ngram_overlap.py <paper.tex> <mold.tex> [n]
"""
import sys

from lexicon_diff import words


def grams(text, n):
    w = [x.lower() for x in words(text)]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    paper = open(sys.argv[1], encoding="utf-8").read()
    mold = open(sys.argv[2], encoding="utf-8").read()
    shared = sorted(grams(paper, n) & grams(mold, n))
    for g in shared:
        print(" ".join(g))
    print(f"\n{len(shared)} shared {n}-grams")


if __name__ == "__main__":
    main()
