#!/usr/bin/env python3
"""Plot error vs benchmark length from results/convergence.csv.

One color and marker per SKU, cold solid and warm dashed, a light band for the range
across instances, a shaded band at the tolerance, and a dotted drop line where each
cold median last crosses the tolerance (the number the memo quotes).

  python plot_convergence.py
  python plot_convergence.py --estimator both --tol 2
"""
import argparse
import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Distinct in hue and in luminance, so the figure survives a grayscale print.
PALETTE = ["#13375c", "#c0561b", "#2f7d53", "#6b4e9b", "#8a6d1f"]
MARKERS = ["o", "s", "^", "D", "v"]
# linestyle, linewidth, filled marker, legend label
STYLES = {"cold": ("-", 1.9, True, "cold (first run)"),
          "warm": ((0, (4, 2.5)), 1.3, False, "warm"),
          "recooled": ((0, (1, 1.7)), 1.6, False, "cold again (after idle)")}


def short(sku):
    s = sku.replace("NVIDIA ", "").replace("Graphics Device", "GPU")
    return s if len(s) <= 22 else s[:21] + "."


def load(path):
    """-> {estimator: {(sku, state): [(t array, err array), ...]}}, one entry per repeat."""
    raw = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("error_pct"):
                continue
            raw[(r["estimator"], r["sku"], r["state"], r["run"], r["repeat"])].append(
                (float(r["seconds"]), float(r["error_pct"])))
    out = defaultdict(lambda: defaultdict(list))
    for (est, sku, state, _run, _rep), pts in raw.items():
        pts.sort()
        if len(pts) < 5:
            continue
        out[est][(sku, state)].append((np.array([p[0] for p in pts]),
                                       np.array([p[1] for p in pts])))
    return out


def envelope(series, grid):
    """Median, min and max across repeats, each interpolated onto a shared time grid."""
    m = np.full((len(series), len(grid)), np.nan)
    for i, (ts, es) in enumerate(series):
        inside = (grid >= ts[0]) & (grid <= ts[-1])
        m[i, inside] = np.interp(grid[inside], ts, es)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(m, axis=0), np.nanmin(m, axis=0), np.nanmax(m, axis=0)


def panel(ax, data, tol, title):
    skus = sorted({sku for sku, _ in data})
    tmax = max(ts[-1] for s in data.values() for ts, _ in s)
    tmin = max(min(ts[0] for s in data.values() for ts, _ in s), 0.9)
    grid = np.logspace(math.log10(tmin), math.log10(tmax), 300)

    ax.axhspan(0, tol, color="0.45", alpha=0.13, lw=0, zorder=0)
    ax.axhline(tol, color="0.4", lw=0.8, ls=":", zorder=1)

    ceiling = tol * 2.5
    for i, sku in enumerate(skus):
        color = PALETTE[i % len(PALETTE)]
        for state in STYLES:
            series = data.get((sku, state))
            if not series:
                continue
            m, lo, hi = envelope(series, grid)
            ls, lw, filled, _ = STYLES[state]
            cold = state == "cold"
            ax.fill_between(grid, lo, hi, color=color, alpha=0.12, lw=0, zorder=2)
            ax.plot(grid, m, color=color, lw=lw, ls=ls, zorder=4,
                    marker=MARKERS[i % len(MARKERS)], markevery=(14 * i + 20, 60), ms=4.2,
                    mfc=color if filled else "white", mec=color, mew=1.1)
            ok = ~np.isnan(m) & (grid > 5)
            if ok.any():
                ceiling = max(ceiling, float(np.nanmax(m[ok])))
            if cold:  # drop line at the last crossing of the tolerance
                above = np.where(~np.isnan(m) & (m > tol))[0]
                if len(above) and above[-1] + 1 < len(grid):
                    x = grid[above[-1] + 1]
                    ax.plot([x, x], [0, tol], color=color, lw=1.0, ls=":", zorder=3)
                    ax.plot([x], [0], color=color, marker="^", ms=4.5, clip_on=False, zorder=5)

    ax.set_xscale("log")
    ax.set_ylim(0, min(ceiling * 1.12, 40))
    ax.set_xlim(tmin, tmax)
    ax.set_xlabel("benchmark length (s, warm-up included)")
    ax.set_ylabel(f"error vs steady state (%)")
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(True, which="major", axis="both", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], color=PALETTE[i % len(PALETTE)], lw=1.9,
                      marker=MARKERS[i % len(MARKERS)], ms=4.2, label=short(sku))
               for i, sku in enumerate(skus)]
    seen = {st for _, st in data}
    handles += [Line2D([], [], color="0.35", lw=lw, ls=ls, label=lab)
                for st, (ls, lw, _f, lab) in STYLES.items() if st in seen]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, ncol=2,
              loc="upper right", handlelength=2.4, columnspacing=1.2)
    return skus


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/convergence.csv")
    p.add_argument("--out", default="results/convergence.png")
    p.add_argument("--tol", type=float, default=2.0)
    p.add_argument("--estimator", default="from_start",
                   help="estimator name from the csv, or 'both' for one panel each")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    by_est = load(args.csv)
    if not by_est:
        raise SystemExit(f"no usable rows in {args.csv}")
    names = sorted(by_est)
    if args.estimator in ("both", "all"):
        picked = names
    elif args.estimator in by_est:
        picked = [args.estimator]
    else:
        raise SystemExit(f"estimator {args.estimator!r} not in {names}")

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(len(picked), 1, figsize=(6.4, 3.5 * len(picked)), squeeze=False)
    for ax, est in zip(axes[:, 0], picked):
        warm = est.replace("warmup", "").replace("s", "")
        label = ("averaging from second one" if est == "from_start"
                 else f"discarding the first {warm} s")
        panel(ax, by_est[est], args.tol, f"{est}: {label}")
    fig.tight_layout(pad=0.9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}  ({', '.join(picked)})")


if __name__ == "__main__":
    main()
