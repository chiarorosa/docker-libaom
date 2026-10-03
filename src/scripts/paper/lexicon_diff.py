"""Lexical parity check: words of a paper that never occur in the mold paper.

Used to keep the ISCAS NPL-AV1 vocabulary inside the LASCAS SNP-AV1 lexicon
(ISCAS/CLAUDE.md, section 3b). Every word listed must be either a new
technical fact of the paper (codec names, function names) or be replaced.

    py src/scripts/paper/lexicon_diff.py <paper.tex> <mold.tex> [extra_text_files...]
"""
import re
import sys


def words(text):
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)   # LaTeX comment, not \%
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.S)
    text = re.sub(r"\\(cite|label|ref|eqref|includegraphics|url)\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return re.findall(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]", text)


def main():
    paper = open(sys.argv[1], encoding="utf-8").read()
    for extra in sys.argv[3:]:
        paper += "\n" + open(extra, encoding="utf-8").read()
    mold = open(sys.argv[2], encoding="utf-8").read()
    mold_set = {w.lower() for w in words(mold)}
    seen = {}
    for w in words(paper):
        lw = w.lower()
        if lw not in mold_set:
            seen[lw] = seen.get(lw, 0) + 1
    for w, n in sorted(seen.items()):
        print(f"{n:3d}  {w}")
    print(f"\n{len(seen)} word types absent from the mold")


if __name__ == "__main__":
    main()
