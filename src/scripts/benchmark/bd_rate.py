#!/usr/bin/env python3
"""Bjontegaard delta-rate (BD-rate) over rate-distortion curves.

BD-rate = average % bitrate difference between two encoders at equal quality,
integrated over the overlapping PSNR range (negative => the test encoder needs
less rate for the same quality => better). Uses monotone piecewise-cubic (PCHIP)
interpolation of log10(rate) as a function of PSNR, the standard formulation.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


def _prep(rate, psnr):
    rate = np.asarray(rate, dtype=np.float64)
    psnr = np.asarray(psnr, dtype=np.float64)
    order = np.argsort(psnr)
    psnr, lr = psnr[order], np.log10(rate[order])
    # PCHIP needs strictly increasing x (psnr); drop duplicates.
    keep = np.concatenate(([True], np.diff(psnr) > 1e-9))
    return psnr[keep], lr[keep]


def bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, samples=100):
    """Percent BD-rate of `test` relative to `anchor`. Needs >=2 overlapping
    points; returns nan if the PSNR ranges do not overlap enough."""
    pa, la = _prep(rate_anchor, psnr_anchor)
    pt, lt = _prep(rate_test, psnr_test)
    if len(pa) < 2 or len(pt) < 2:
        return float("nan")
    lo = max(pa.min(), pt.min())
    hi = min(pa.max(), pt.max())
    if hi <= lo:
        return float("nan")
    fa = PchipInterpolator(pa, la)
    ft = PchipInterpolator(pt, lt)
    xs = np.linspace(lo, hi, samples)
    avg_log_diff = np.mean(ft(xs) - fa(xs))
    return float((10.0 ** avg_log_diff - 1.0) * 100.0)


if __name__ == "__main__":
    # Self-check: identical curves -> ~0 BD-rate.
    r = [100.0, 200.0, 400.0, 800.0]
    q = [34.0, 37.0, 40.0, 43.0]
    print("identical:", round(bd_rate(r, q, r, q), 4))
    # A uniformly cheaper test encoder -> negative BD-rate.
    print("10% cheaper:", round(bd_rate(r, q, [x * 0.9 for x in r], q), 4))
