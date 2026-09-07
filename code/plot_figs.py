"""plot_figs.py — draw the three main-text figures from results/*.csv.

Kept separate from make_figs.py so the figures can be redrawn without rerunning
the simulations.  Rules followed here (PLAN §8, HSSC + user 2026-09-06):
  * panel labels lowercase bold (a, b, c), title text after them;
  * no text on top of a line or a bar; labels sit in empty regions, and every
    annotation points at the thing it names;
  * a legend whenever a panel has two or more series, placed outside the data;
  * no dashes in any label; plain sentences;
  * 300 dpi PNG plus a vector PDF of each figure; nothing clipped (bbox tight).
Usage: python3 code/plot_figs.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import interval as iv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GREY1, GREY2 = "#9a9891", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def read(name):
    with open(os.path.join(RES, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def ci(p, n):
    return 1.96 * np.sqrt(np.maximum(p * (1 - p), 1e-12) / n)


def panel(ax, letter, title):
    """HSSC style: lowercase bold panel letter, then the title in normal weight."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10, pad=6)
    ax.set_title(title, loc="center", fontsize=9, pad=6)


def save(fig, stem):
    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig1():
    rows = read("fig1_trajectory.csv")
    x = np.array([r["x"] for r in rows]); steps = np.arange(1, len(x) + 1)
    L = np.cumsum(2 * x / iv.SIGMA ** 2)
    up_c, _ = iv.thresholds_for(iv.CALIBRATED); up_o, _ = iv.thresholds_for(iv.OVERCONFIDENT)
    thr_c, thr_o = up_c[1], up_o[1] / 2.0
    lp = 2.0 * L
    t_o = int(np.argmax(lp >= up_o[1]) + 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.grid(False)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(thr_c, color=BLUE, lw=1.3, ls="--")
    ax.axhline(thr_o, color=ORANGE, lw=1.3, ls="--")
    ax.axvspan(t_o, t_o + 2, color=GRID, alpha=.75, lw=0)
    ax.plot(steps, L, color=INK, lw=2, marker="o", ms=4.5, zorder=4)
    ax.scatter([t_o], [L[t_o - 1]], s=170, facecolor="none", edgecolor=ORANGE, lw=1.8, zorder=5)
    xr = len(L) + 0.45
    # bar labels: right end, above each line, where the trajectory is far below
    ax.text(xr, thr_c + 0.15, "the calibrated person acts above this line", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.text(xr, thr_o + 0.15, "the overconfident person acts above this line\n(each observation counted twice)", ha="right", va="bottom", fontsize=7.5, color=INK2)
    # the act: text in the empty band between 0 and the orange bar, with a leader to the circled point
    ax.annotate("step %d: the overconfident person acts.\nThe action is final, and the state is bad." % t_o,
                xy=(t_o + 0.12, L[t_o - 1] - 0.15), xytext=(t_o + 1.35, 1.55), va="top", ha="left", fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=2))
    # the interval: text inside the shaded band, below the trajectory
    ax.text(t_o + 0.1, min(L) + 1.4, "an imposed wait of two more steps\n(shaded): the run breaks before\nanyone may act", ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlabel("step (one observation per step)")
    ax.set_ylabel("evidence that the state is good\n(log-odds; 0 means even)")
    ax.set_xlim(0.5, xr); ax.set_xticks(steps); ax.set_ylim(min(L) - 0.5, thr_c + 1.3)
    save(fig, "fig1_phantom")
    print("fig1 drawn; overconfident acts at step", t_o)


# ---------------------------------------------------------------- Fig 2
def fig2():
    rows = read("fig2.csv")
    N = rows[0]["n"]
    gammas = [1.0, 1.5, 2.0, 3.0]; ks = sorted({int(r["k"]) for r in rows})
    cols = {1.0: BLUE, 1.5: AQUA, 2.0: ORANGE, 3.0: YELLOW}
    names = {1.0: "calibrated (γ = 1)", 1.5: "mildly overconfident (γ = 1.5)", 2.0: "overconfident (γ = 2)", 3.0: "very overconfident (γ = 3)"}
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.4))
    handles = []
    for g in gammas:
        sub = sorted([r for r in rows if r["gamma"] == g], key=lambda r: r["k"])
        k = [r["k"] for r in sub]
        for ax, key in zip(axes, ("avoidable", "missed", "payoff")):
            y = np.array([r[key] for r in sub])
            if key != "payoff":
                e = ci(y, N); ax.fill_between(k, 100 * (y - e), 100 * (y + e), color=cols[g], alpha=.15, lw=0)
                h, = ax.plot(k, 100 * y, color=cols[g], lw=2, marker="o", ms=4)
            else:
                e = np.array([r["payoff_se"] for r in sub]) * 1.96
                ax.fill_between(k, y - e, y + e, color=cols[g], alpha=.15, lw=0)
                h, = ax.plot(k, y, color=cols[g], lw=2, marker="o", ms=4)
            if key == "avoidable":
                handles.append((h, names[g]))
    # luck floor for the overconfident person: failures after acting in the good state
    sub2 = sorted([r for r in rows if r["gamma"] == 2.0], key=lambda r: r["k"])
    floor = [100 * (r["wrong"] - r["avoidable"]) for r in sub2]
    hf, = axes[0].plot(ks, floor, color=ORANGE, lw=1.2, ls="--")
    axes[0].annotate("failures after acting in the good state\n(overconfident person). Waiting leaves these.",
                     xy=(ks[-2], floor[-2]), xytext=(ks[-1] + 0.05, floor[-1] + 1.0), ha="right", va="bottom",
                     fontsize=6.8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=2))
    panel(axes[0], "a", "avoidable catastrophes\n(acted in the bad state)")
    panel(axes[1], "b", "missed opportunities\n(did not act in the good state)")
    panel(axes[2], "c", "average payoff")
    for ax in axes:
        ax.set_xlabel("imposed wait before any decision (steps)"); ax.set_xticks(ks)
    axes[0].set_ylabel("% of decisions"); axes[1].set_ylabel("% of decisions"); axes[2].set_ylabel("payoff units")
    axes[0].set_ylim(0, max(100 * r["avoidable"] for r in rows) * 1.4)
    fig.legend([h for h, _ in handles] + [hf], [n for _, n in handles] + ["luck floor (see panel a)"],
               loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig2_waiting")
    print("fig2 drawn")


# ---------------------------------------------------------------- Fig 3
def fig3():
    rows = read("fig3.csv")

    def pick(person, device, verdict):
        return [r for r in rows if r["person"] == person and r["device"] == device and r["verdict"] == verdict][0]

    cal = pick("calibrated", "none", "all")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, person, letter, title in zip(axes, ("anxious", "fearful"), "ab",
                                         ("the anxious person (waiting hurts)", "the fearful person (failure brings blame)")):
        none = pick(person, "none", "all"); a0 = pick(person, "verdict theta=0.0", "all")
        w9 = pick(person, "verdict theta=0.9", "wait"); g9 = pick(person, "verdict theta=0.9", "go")
        cells = [none, a0, w9, g9]
        xs = [0, 1, 2.0, 2.6]; colors = [GREY1, GREY2, BLUE, ORANGE]
        for x, r, c in zip(xs, cells, colors):
            v, n = r["missed"], r["n"]
            ax.bar(x, 100 * v, width=0.56, color=c, lw=0)
            ax.errorbar(x, 100 * v, yerr=100 * ci(v, n), color=INK, lw=0.8, capsize=2)
            ax.text(x, 100 * v + 100 * ci(v, n) + 0.15, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
        hcal = ax.axhline(100 * cal["missed"], color=INK2, lw=1, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(["no\ndevice", "random verdict,\nno relief\n(θ = 0)", "drew\n\"wait\"", "drew\n\"go\""], fontsize=7.5)
        ax.plot([1.72, 2.88], [-3.2, -3.2], color=INK2, lw=0.8, clip_on=False)
        ax.text(2.3, -3.5, "random verdict with relief (θ = 0.9)", ha="center", va="top", fontsize=7.5, color=INK, clip_on=False)
        ax.set_xlim(-0.55, 3.6); ax.set_ylim(0, 9)
        ax.grid(axis="x", visible=False)
        panel(ax, letter, title)
    axes[0].set_ylabel("missed opportunities\n(% of decisions)")
    fig.legend([hcal], [f"dashed line: the calibrated person, {100*cal['missed']:.1f}%"], loc="upper right",
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.99, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig3_two_sided")
    print("fig3 drawn")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
