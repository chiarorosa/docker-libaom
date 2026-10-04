"""Audit of every number of the ISCAS NPL-AV1 paper against the experimental artifacts.

Every expected value is recomputed here from the CSV artifacts (or taken from the
canonical thesis documents when no artifact exists, flagged as such). Every number
of the paper body (abstract to conclusions, tables included) must match an
expected value; unmatched numbers are listed for manual review.

Run from the repository root:
    py src/scripts/paper/audit_numbers_iscas.py
"""
import csv
import re
import statistics as st

B = "results/benchmark/"
TEX = "results/thesis/IEEE_Conference_Template/ISCAS/PAPER_ISCAS_2027_NPL-AV1.tex"


def rows(p):
    return list(csv.DictReader(open(B + p)))


exp = {}  # value string -> provenance


def add(v, nd, prov):
    exp.setdefault(("%." + str(nd) + "f") % v, set()).add(prov)


sw = {(r["level"], r["kind"]): r for r in rows("fase6_swap/swap_average.csv")}
sc = {(r["level"], r["kind"]): r for r in rows("fase6_swap_h9c/swap_average.csv")}
for (lv, k), r in list(sw.items()) + list(sc.items()):
    add(float(r["bd_rate"]), 3, "TabI %s p%s BD" % (k, lv))
    add(float(r["ts_pct"]), 2, "TabI %s p%s TS" % (k, lv))
    add(float(r["speedup"]), 2, "speedup %s p%s" % (k, lv))
d2 = {r["cpu"]: r for r in rows("fase6_swap_h9d/marginal_average.csv")}
for c, r in d2.items():
    add(float(r["h9d_bd"]), 3, "TabI ext p%s BD" % c)
    add(float(r["h9d_bd"]), 2, "ext p%s BD (2 casas)" % c)
    add(float(r["h9d_ts"]), 2, "TabI ext p%s TS" % c)
    add(float(r["dts"]), 2, "ext dTS p%s" % c)
    add(float(r["dbd"]), 3, "ext dBD p%s" % c)
    add(float(r["dts_p"]), 3, "ext dTS p p%s" % c)
for r in rows("fase6_swap_h9d/marginal_per_seq.csv"):
    add(float(r["dts"]), 2, "TabIII ext dTS %s p%s" % (r["seq"], r["cpu"]))
for r in rows("fase6_swap_h9c/swap_per_seq.csv"):
    if r["level"] == "1" and r["kind"] in ("native", "h9c_tau95"):
        add(float(r["bd_rate"]), 3, "TabIII %s %s BD" % (r["seq"], r["kind"]))
        add(float(r["ts_pct"]), 2, "TabIII %s %s TS" % (r["seq"], r["kind"]))
for r in rows("fase6_analysis/paired_tests.csv"):
    if r["config"].startswith("h9c_tau95"):
        nd = 3 if r["metric"] == "bd_rate" else 2
        add(float(r["mean_delta"]), nd, "TabII %s %s %s" % (r["config"], r["regime"], r["metric"]))
        add(float(r["p"]), 3, "TabII p %s %s %s" % (r["config"], r["regime"], r["metric"]))
fr = {r["config"]: r for r in rows("fase6_analysis/pareto_frontier.csv")}
add(float(fr["h9c_tau95"]["bd_rate"]), 3, "p0 point BD")
add(float(fr["h9c_tau95"]["ts_pct"]), 2, "p0 point TS")
# derived ratios of the text
ef = (float(d2["1"]["dts"]) / float(d2["1"]["dbd"]), float(d2["2"]["dts"]) / float(d2["2"]["dbd"]))
add(ef[0], 0, "ext efficiency p1"); add(ef[1], 0, "ext efficiency p2")
for lv in ("1", "2"):
    k = ((float(sw[(lv, "h9a_aggr")]["ts_pct"]) - float(sw[(lv, "h9a_bal")]["ts_pct"]))
         / (float(sw[(lv, "h9a_aggr")]["bd_rate"]) - float(sw[(lv, "h9a_bal")]["bd_rate"])))
    add(k, 1, "threshold knob price p%s" % lv)
for lv in ("1", "2", "3"):
    add(float(sw[(lv, "h9a_bal")]["bd_rate"]) / float(sw[(lv, "native")]["bd_rate"]), 1,
        "BD ratio eff-first/native p%s" % lv)
ov = rows("overhead_iscas/overhead.csv")
for r in ov:
    w = float(r["wall_s"]) * 1000
    for k, ks in (("cnn", ["native_cnn"]), ("pre", ["h9a_infer", "h9a_feature_prep"]),
                  ("post", ["h9c_infer", "h9c_feature_prep"]), ("ext", ["h9d_infer", "h9d_feature_prep"])):
        v = 100 * sum(float(r[x + "_ms"]) for x in ks) / w
        if v > 0:
            add(v, 2, "overhead %s %s %s" % (k, r["seq"], r["config"]))
# canonical documents (no CSV in this audit)
for v, prov in [("0.23", "M3 3.6 repeat SD"), ("0.46", "M3 resolution"), ("1.9", "R3 3.5 interaction"),
                ("0.0910", "theta16"), ("0.1031", "theta32"), ("0.0144", "theta64"),
                ("0.95", "tau"), ("0.90", "tau"), ("0.20", "tau"), ("0.60", "tau"), ("0.85", "tau"),
                ("0.40", "tau"), ("0.92", "eff-first p1 BD rounded"), ("4.35", "time-first p3 BD rounded"),
                ("0.41", "post-NONE p1 BD rounded"), ("1.07", "ext p2 BD rounded"), ("0.04", "dBD bound")]:
    exp.setdefault(v, set()).add(prov)

t = open(TEX, encoding="utf-8").read()
body = t.split("\\begin{abstract}")[1].split("\\section*{Acknowledgment}")[0]
body = re.sub(r"(?m)(?<!\\)%.*$", "", body)
body = re.sub(r"\\cite\{[^}]*\}|\\label\{[^}]*\}|\\ref\{[^}]*\}|\\includegraphics\[[^]]*\]\{[^}]*\}", " ", body)
nums = re.findall(r"(?<![\w.])[-+]?\d+\.\d+", body)
ok, bad = 0, []
for n in nums:
    key = n.lstrip("+").replace("-", "")
    cand = {key, "-" + key}
    if any(c in exp for c in cand):
        ok += 1
    else:
        bad.append(n)
print("numeros decimais no corpo:", len(nums), "| casados:", ok, "| sem fonte:", len(bad))
for b in sorted(set(bad)):
    print("  SEM FONTE:", b)
