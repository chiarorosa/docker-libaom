import csv, os
base="/workspace/results/benchmark/h9_test"
SEQS=["Jockey","RaceNight","RiverBank"]
def curve(seq):
    pts=[]
    for sub in ("curve_safe","curve_aggr"):
        p=f"{base}/{seq}/{sub}/summary.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if "H8" in r["point"]: continue
                pts.append((r["point"],float(r["bd_rate_pct"]),float(r["time_speedup_x"]),float(r["ts_pct"])))
    return pts
def abl(seq,m):
    p=f"{base}/{seq}/ablation/curve.csv"; pts=[]
    for r in csv.DictReader(open(p)):
        if r["method"]==m: pts.append((float(r["bd_rate_pct"]),float(r["speedup_x"]),float(r["ts_pct"])))
    return sorted(pts,key=lambda x:x[1])
def interp_bd_at_speed(pts,s):
    xs=[p[1] for p in pts]
    if s<xs[0] or s>xs[-1]: return None
    for i in range(1,len(pts)):
        b0,s0,_=pts[i-1]; b1,s1,_=pts[i]
        if s0<=s<=s1: f=0 if s1==s0 else (s-s0)/(s1-s0); return b0+f*(b1-b0)
for seq in SEQS:
    print(f"\n########## {seq} ##########")
    print(" H9a deployable curve (BD% | TS% | speedup):")
    for n,bd,sp,ts in curve(seq): print(f"   {n:<16} {bd:6.3f} | {ts:5.1f} | {sp:.3f}x")
    ml=abl(seq,"ml"); va=abl(seq,"variance"); rn=abl(seq,"random")
    print(f" ablation ml speedup range:   {ml[0][1]:.3f}..{ml[-1][1]:.3f}x  (BD {ml[0][0]:.3f}..{ml[-1][0]:.3f}%)")
    print(f" ablation var speedup range:  {va[0][1]:.3f}..{va[-1][1]:.3f}x  (BD {va[0][0]:.3f}..{va[-1][0]:.3f}%)")
    ov_lo=max(ml[0][1],va[0][1]); ov_hi=min(ml[-1][1],va[-1][1])
    print(f" ml/variance speedup OVERLAP: {'YES '+f'{ov_lo:.2f}..{ov_hi:.2f}x' if ov_lo<=ov_hi else 'NONE'}")
    print(f" matched-POLICY attribution: ml BD floor {ml[0][0]:.3f}%  vs  variance BD floor {va[0][0]:.3f}%")
    # ml vs random at matched speedup (grid)
    print(" ml vs random @ matched speedup (BD%):")
    for s in [1.10,1.15,1.30,1.45]:
        m=interp_bd_at_speed(ml,s); r=interp_bd_at_speed(rn,s)
        if m is not None and r is not None:
            print(f"   {s:.2f}x: ml {m:.3f}  random {r:.3f}  -> {'ml' if m<r else 'random'}")
