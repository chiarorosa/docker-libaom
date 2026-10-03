"""Stylistic profile of a paper body (Introduction..Acknowledgment).

Used to keep the ISCAS NPL-AV1 text in parity with the LASCAS SNP-AV1 mold:
    py src/scripts/paper/style_profile.py <paper.tex>
"""
import re, collections, sys
t = open(sys.argv[1], encoding="utf-8").read()
body = t[t.find(r"\section{Introduction}"):t.find(r"\section*{Acknowledgment}")]
body = re.sub(r"(?m)%.*$", "", body)
body = re.sub(r"\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", "", body, flags=re.S)
body = re.sub(r"\\(sub)?section\{[^}]*\}|\\label\{[^}]*\}", " ", body)
body = re.sub(r"\\cite\{[^}]*\}", "[C]", body)
body = re.sub(r"\$[^$]*\$", "X", body)
body = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", body)
body = re.sub(r"\s+", " ", body)
sents = [s.strip() for s in re.split(r"(?<=[.])\s+(?=[A-Z])", body) if len(s.split()) > 3]
L = [len(s.split()) for s in sents]
print("sentences", len(sents), "words", sum(L), "mean", round(sum(L)/len(L), 1),
      "median", sorted(L)[len(L)//2], "max", max(L))
print("share >30w", round(sum(l > 30 for l in L)/len(L), 2), " <15w", round(sum(l < 15 for l in L)/len(L), 2))
low = body.lower()
pats = {"rather than": r"rather than", "', not' / 'and not'": r",? (and )?not (the|a|by|from|of|in|on|to|as|only)\b",
        "colon": r":", "semicolon": r";", ", so": r", so ", "since": r"\bsince\b", "because": r"\bbecause\b",
        "whenever": r"\bwhenever\b", "however": r"\bhowever\b", "which is why": r"which is why",
        "by construction": r"by construction", "at the price/cost of": r"at (the|a) (price|cost) of",
        "passive be+ed": r"\b(is|are|was|were|be|been)\s+\w+ed\b", "this paper/work": r"this (paper|work)",
        "we/our": r"\b(we|our)\b", "may be/likely": r"\b(may be|likely)\b", "(i)": r"\(i\)",
        "verified": r"verif", "measured": r"measured", "one ... follows": r"follows"}
for k, p in pats.items():
    print(f"{k:28s}{len(re.findall(p, low))}")
print(collections.Counter(" ".join(s.split()[:2]) for s in sents).most_common(15))
for s in sents:
    if re.search(r"rather than|, not |and not |which is why|follows|by construction", s.lower()):
        print(" >", s[:170])
