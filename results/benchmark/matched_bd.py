import csv, os
base="/workspace/results/benchmark/h9_test"
def load_curve(seq):
    pts=[]
    for sub in ("curve_safe","curve_aggr"):
        p=f"{base}/{seq}/{sub}/summary.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if "H8" in r["point"]: continue
                pts.append((float(r["bd_rate_pct"]),float(r["time_speedup_x"]),float(r["ts_pct"])))
    return sorted(pts)
def load_abl(seq,method):
    p=f"{base}/{seq}/ablation/curve.csv"; pts=[]
    if not os.path.exists(p): return []
    for r in csv.DictReader(open(p)):
        if r["method"]==method:
            pts.append((float(r["bd_rate_pct"]),float(r["speedup_x"]),float(r["ts_pct"])))
    return sorted(pts)
def interp(pts, bd):
    xs=[p[0] for p in pts]
    if bd<xs[0] or bd>xs[-1]: return None
    for i in range(1,len(pts)):
        x0,s0,t0=pts[i-1]; x1,s1,t1=pts[i]
        if x0<=bd<=x1:
            f=0 if x1==x0 else (bd-x0)/(x1-x0); return s0+f*(s1-s0), t0+f*(t1-t0)
for seq in ("Jockey","RaceNight"):
    ml_dep=load_curve(seq); ml_abl=load_abl(seq,"ml"); var=load_abl(seq,"variance")
    if not var: continue
    print(f"\n===== {seq} =====")
    print(f"  ml (ablation NONE-only) BD: {ml_abl[0][0]:.2f}..{ml_abl[-1][0]:.2f}%")
    print(f"  ml (deployed +rect-off) BD: {ml_dep[0][0]:.2f}..{ml_dep[-1][0]:.2f}%")
    print(f"  variance BD:                {var[0][0]:.2f}..{var[-1][0]:.2f}%")
    lo=max(ml_dep[0][0], var[0][0]); hi=min(ml_dep[-1][0], var[-1][0])
    if lo>hi:
        print("  -> NO BD overlap even with deployed ml.")
    else:
        print(f"  -> BD overlap (deployed ml vs variance): {lo:.2f}..{hi:.2f}%")
        for bd in [lo,(lo+hi)/2,hi]:
            m=interp(ml_dep,bd); v=interp(var,bd)
            if m and v:
                print(f"     @BD={bd:.2f}%: deployed-ml {m[0]:.2f}x (TS {m[1]:.1f}%) | variance {v[0]:.2f}x (TS {v[1]:.1f}%)")
