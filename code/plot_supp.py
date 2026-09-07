"""plot_supp.py — redraw the four supplementary figures from results/s*.csv.
Same drawing rules as plot_figs.py.  HSSC naming: Supplementary Fig. S1–S4
(S1 = stakes sweep S-A, S2 = p sweep S-B, S3 = θ curve S-D, S4 = pressure strength S-F).
Usage: python3 code/plot_supp.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_figs import read, panel, save, BLUE, ORANGE, INK, INK2, FIGS

CAL_MISSED = None  # the calibrated person's missed rate, read from the p-sweep file (p is irrelevant for them)


def cal_missed():
    global CAL_MISSED
    if CAL_MISSED is None:
        CAL_MISSED = 100 * float(np.mean([r["missed_calibrated"] for r in read("sB_p.csv")]))
    return CAL_MISSED


def s1_stakes():
    rows = sorted(read("sA_stakes.csv"), key=lambda r: r["step_over_catastrophe"])
    x = [100 * r["step_over_catastrophe"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(x, [r["overconf_d_payoff"] for r in rows], color=ORANGE, s=34, marker="s", label="overconfident person", zorder=3)
    ax.scatter(x, [r["d_payoff"] for r in rows], color=INK, s=34, label="population (four kinds in equal shares)", zorder=3)
    ax.scatter(x, [r["calibrated_d_payoff"] for r in rows], color=BLUE, s=34, marker="^", label="calibrated person", zorder=3)
    ax.set_xscale("log"); ticks = [v for v in x if abs(v - 1 / 3) > 1e-3]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks]); ax.minorticks_off()
    ax.set_xlabel("cost of one step of waiting, as % of the catastrophe (log scale)")
    ax.set_ylabel("payoff of the whole procedure\nminus payoff of own judgement")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_ylim(min(r["calibrated_d_payoff"] for r in rows) - 0.55, max(r["overconf_d_payoff"] for r in rows) + 0.15)
    panel(ax, "", "the interval pays when a step of waiting is cheap against the catastrophe")
    save(fig, "supp_S1_stakes")


def s2_p():
    rows = sorted(read("sB_p.csv"), key=lambda r: r["p"])
    ps = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(ps, [100 * r["missed_anxious"] for r in rows], color=BLUE, marker="o", lw=2, label="anxious person")
    ax.plot(ps, [100 * r["missed_fearful"] for r in rows], color=ORANGE, marker="o", lw=2, label="fearful person")
    ax.plot(ps, [100 * r["pop_missed"] for r in rows], color=INK, marker="o", lw=2, ls="--", label="population (equal shares)")
    ax.set_xlabel("probability that the verdict says \"go\"\n(0 means it always says \"wait\"; 1 means it always says \"go\")")
    ax.set_ylabel("missed opportunities (% of decisions)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    panel(ax, "", "outcomes are straight lines in p: the draw itself adds nothing in expectation")
    save(fig, "supp_S2_p")


def s3_theta():
    rows = sorted(read("sD_theta.csv"), key=lambda r: r["theta"])
    th = [r["theta"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(th, [100 * r["anxious_wait_missed"] for r in rows], color=BLUE, marker="o", lw=2, label="anxious person who drew \"wait\"")
    ax.plot(th, [100 * r["fearful_go_missed"] for r in rows], color=ORANGE, marker="o", lw=2, label="fearful person who drew \"go\"")
    ax.axhline(cal_missed(), color=INK2, ls="--", lw=1, label=f"calibrated person ({cal_missed():.1f}%)")
    ax.set_xlabel("θ, the share of the pressure that the verdict lifts")
    ax.set_ylabel("missed opportunities (% of decisions)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    panel(ax, "", "the correction grows with the relief; full relief reaches the calibrated rate")
    save(fig, "supp_S3_theta")


def s4_pressure():
    rows = read("sF_pressure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, person, col, letter, xlabel in ((axes[0], "anxious", BLUE, "a", "extra cost of each step of waiting, C_anx"),
                                            (axes[1], "fearful", ORANGE, "b", "extra cost of blame after a failure, R_blame")):
        sub = sorted([r for r in rows if r["person"] == person], key=lambda r: r["strength"])
        xs = [r["strength"] for r in sub]
        ax.plot(xs, [100 * r["none_missed"] for r in sub], color=col, marker="o", lw=2, label=f"{person} person, alone")
        ax.plot(xs, [100 * r["relieved_missed"] for r in sub], color=col, marker="o", lw=2, ls="--", label=f"{person} person after the relieving verdict (θ = 0.9)")
        ax.axhline(100 * sub[0]["calibrated_missed"], color=INK2, ls=":", lw=1, label="calibrated person")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.minorticks_off()
        ax.set_xlabel(xlabel); ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=3.2, markerscale=0.7)
        panel(ax, letter, f"the {person} person")
    axes[0].set_ylabel("missed opportunities (% of decisions)")
    fig.tight_layout()
    save(fig, "supp_S4_pressure")


if __name__ == "__main__":
    s1_stakes(); s2_p(); s3_theta(); s4_pressure()
    for old in ("supp_sA.png", "supp_sB.png", "supp_sD.png", "supp_sF.png"):
        p = os.path.join(FIGS, old)
        if os.path.exists(p):
            os.remove(p)
    print("supplementary figures drawn: S1–S4")
