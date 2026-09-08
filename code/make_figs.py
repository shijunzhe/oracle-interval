"""make_figs.py — produce results/*.csv and figs/*.png for the main text (M2).

python3 code/make_figs.py            # n=100,000 per cell
"""
# NOTE (2026-09-06): the figures this script draws are superseded by code/plot_figs.py, which redraws them
# from results/*.csv under the figure rules in PLAN §8 (no overlaps, legends outside data, lowercase bold
# panel labels, 300 dpi + PDF). Run this script for the numbers, then `python3 code/plot_figs.py` for the figures.
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import interval as iv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
os.makedirs(RES, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
P_GO = 0.25   # probability that the verdict says "go": one line in four changes under either casting method

# palette (dataviz reference instance, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ================================================================ Fig 1: phantom pattern
def fig1():
    """A representative episode: bad state, a run of favourable evidence, the
    overconfident person acts at their threshold while the calibrated person is
    still waiting, then the evidence turns."""
    up_c, _ = iv.thresholds_for(iv.CALIBRATED)
    up_o, _ = iv.thresholds_for(iv.OVERCONFIDENT)
    rng = np.random.default_rng(3)
    best = None
    for trial in range(20000):
        x = -1 + iv.SIGMA * rng.standard_normal(8)          # bad state
        L = np.cumsum(2 * x / iv.SIGMA ** 2)
        lp = 2.0 * L
        t_o = np.argmax(lp >= up_o[1]) + 1 if (lp >= up_o[1]).any() else None
        # The overconfident person must act at step 2, inside the two imposed steps of
        # Fig. 2 (forced_wait k=3: no decision at steps 1 and 2, first decision at step 3),
        # and by step 3 the perceived evidence must be back under their bar, so that the
        # same rule that Fig. 2 scores would have held this person past the run.
        if t_o is None or t_o != 2:
            continue
        # calibrated must not have acted by t_o, and evidence must later reverse below 0
        if (L[:t_o] >= up_c[1]).any():
            continue
        if L[t_o:].min() > -0.5 or L[t_o + 1] > L[t_o - 1]:
            continue
        if (lp[2:] >= up_o[1]).any() or L.min() < -7.5:
            continue
        score = -L[t_o + 1]  # prefer a clear break at step 3
        if best is None or score > best[0]:
            best = (score, x, L, lp, t_o)
    _, x, L, lp, t_o = best
    steps = np.arange(1, len(L) + 1)
    b_true, b_perc = iv.sigmoid(L), iv.sigmoid(lp)
    thr_c = up_c[1]                 # calibrated bar, in true-evidence units
    thr_o = up_o[1] / 2.0           # overconfident bar, in true-evidence units (he counts evidence twice)

    fig, ax = plt.subplots(figsize=(5.4, 3.3), dpi=200)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(thr_c, color=BLUE, lw=1.2, ls="--")
    ax.axhline(thr_o, color=ORANGE, lw=1.2, ls="--")
    ax.text(0.6, thr_c + 0.12, "bar of the calibrated person", color=BLUE, va="bottom", fontsize=7.5)
    ax.text(0.6, thr_o - 0.12, "bar of the overconfident person (each observation counts double)", color=ORANGE, va="top", fontsize=7.5)
    ax.axvspan(t_o, t_o + 2, color=GRID, alpha=.7, lw=0)
    ax.plot(steps, L, color=INK, lw=2, marker="o", ms=4.5)
    ax.scatter([t_o], [L[t_o - 1]], s=150, facecolor="none", edgecolor=ORANGE, lw=1.8, zorder=5)
    ax.annotate("step %d: the overconfident person acts.\nIrreversible — and the state is bad." % t_o,
                (t_o, L[t_o - 1]), xytext=(t_o + 2.4, thr_o - 0.55), va="top", fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("an imposed interval of two steps:\nthe run breaks before anyone acts", (t_o + 1.0, -2.6),
                xytext=(t_o + 2.4, -2.9), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("step (one observation per step)")
    ax.set_ylabel("accumulated evidence that the state is good\n(log-odds; 0 = even)")
    ax.set_xlim(0.5, len(L) + 0.5); ax.set_xticks(steps); ax.set_ylim(min(L) - 0.6, thr_c + 0.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_phantom.png")); plt.close(fig)
    write_csv(os.path.join(RES, "fig1_trajectory.csv"),
              [dict(step=int(s), x=float(xx), belief_true=float(bt), belief_perceived=float(bp))
               for s, xx, bt, bp in zip(steps, x, b_true, b_perc)])
    print("fig1: overconfident acts at step", t_o)


# ================================================================ Fig 2: waiting helps only the miscalibrated
def fig2():
    gammas = [1.0, 1.5, 2.0, 3.0]; ks = list(range(0, 6))
    rows = []
    for g in gammas:
        P = iv.Person(gamma=g, label=f"gamma={g}")
        for k in ks:
            r = iv.simulate(P, "forced_wait", n=N, k=k + 1, seed=11)
            rows.append(r.row(gamma=g, k=k))
    write_csv(os.path.join(RES, "fig2.csv"), rows)
    cols = {1.0: BLUE, 1.5: AQUA, 2.0: ORANGE, 3.0: YELLOW}
    names = {1.0: "calibrated (γ=1)", 1.5: "mildly overconfident (γ=1.5)", 2.0: "overconfident (γ=2)", 3.0: "very overconfident (γ=3)"}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), dpi=200)
    for g in gammas:
        sub = [r for r in rows if r["gamma"] == g]
        k = [r["k"] for r in sub]
        for ax, key in zip(axes, ("avoidable", "missed", "payoff")):
            y = np.array([r[key] for r in sub])
            if key != "payoff":
                e = ci(y, N); ax.fill_between(k, 100 * (y - e), 100 * (y + e), color=cols[g], alpha=.15, lw=0)
                ax.plot(k, 100 * y, color=cols[g], lw=2, marker="o", ms=4, label=names[g])
            else:
                e = np.array([r["payoff_se"] for r in sub]) * 1.96
                ax.fill_between(k, y - e, y + e, color=cols[g], alpha=.15, lw=0)
                ax.plot(k, y, color=cols[g], lw=2, marker="o", ms=4, label=names[g])
    # luck floor for gamma=2: failures in the good state
    sub2 = [r for r in rows if r["gamma"] == 2.0]
    floor = [100 * (r["wrong"] - r["avoidable"]) for r in sub2]
    axes[0].plot(ks, floor, color=ORANGE, lw=1.2, ls="--")
    axes[0].text(ks[-1], floor[-1] + 0.08, "γ=2: failures of well-founded\nactions (luck) — waiting cannot\nremove these", ha="right", va="bottom", fontsize=6.8, color=ORANGE)
    axes[0].set_title("A  avoidable catastrophes\n(acted in the bad state)", loc="left", fontsize=9)
    axes[1].set_title("B  missed opportunities\n(did not act in the good state)", loc="left", fontsize=9)
    axes[2].set_title("C  average payoff", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("imposed wait before any decision (steps)"); ax.set_xticks(ks)
    axes[0].set_ylabel("% of decisions"); axes[1].set_ylabel("% of decisions"); axes[2].set_ylabel("payoff units")
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_waiting.png")); plt.close(fig)
    r0 = [r for r in sub2 if r["k"] == 0][0]; r2 = [r for r in sub2 if r["k"] == 2][0]; r3 = [r for r in sub2 if r["k"] == 3][0]
    print(f"fig2: γ=2 avoidable {100*r0['avoidable']:.2f}% -> k=2 {100*r2['avoidable']:.2f}% ({100*(1-r2['avoidable']/r0['avoidable']):.0f}% cut) -> k=3 {100*r3['avoidable']:.2f}%; payoff {r0['payoff']:.2f} -> {r2['payoff']:.2f}")


# ================================================================ Fig 3: one draw, two directions
def fig3():
    rows = []
    out = {}
    for P in (iv.ANXIOUS, iv.FEARFUL, iv.CALIBRATED):
        r_none = iv.simulate(P, "none", n=N, seed=23); rows.append(r_none.row(person=P.label, device="none", verdict="all"))
        for th in (0.0, 0.9):
            allr, gor, wtr = iv.simulate(P, "verdict", n=N, seed=23, theta=th, p=P_GO, split=True)
            rows.append(allr.row(person=P.label, device=f"verdict theta={th}", verdict="all"))
            rows.append(gor.row(person=P.label, device=f"verdict theta={th}", verdict="go"))
            rows.append(wtr.row(person=P.label, device=f"verdict theta={th}", verdict="wait"))
            out[(P.label, th)] = (allr, gor, wtr)
        out[(P.label, "none")] = r_none
    write_csv(os.path.join(RES, "fig3.csv"), rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), dpi=200, sharey=True)
    cal = out[("calibrated", "none")]
    for ax, P, title in zip(axes, (iv.ANXIOUS, iv.FEARFUL),
                            ("A  the anxious person (waiting hurts)", "B  the fearful person (failure brings blame)")):
        none = out[(P.label, "none")]; a0 = out[(P.label, 0.0)][0]; a9, g9, w9 = out[(P.label, 0.9)]
        vals = [none.missed, a0.missed, w9.missed, g9.missed]; ns = [none.n, a0.n, w9.n, g9.n]
        xs = [0, 1, 2.0, 2.6]; colors = ["#9a9891", "#c3c2b7", BLUE, ORANGE]
        for x, v, c, nn in zip(xs, vals, colors, ns):
            ax.bar(x, 100 * v, width=0.56, color=c, lw=0)
            ax.errorbar(x, 100 * v, yerr=100 * ci(v, nn), color=INK, lw=0.8, capsize=2)
            ax.text(x, 100 * v + 0.4, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.axhline(100 * cal.missed, color=INK2, lw=1, ls="--")
        ax.text(3.0, 100 * cal.missed, "calibrated\nperson", fontsize=7, color=INK2, va="center", ha="left")
        ax.set_xticks([0, 1, 2.0, 2.6])
        ax.set_xticklabels(["no\ndevice", "random verdict,\nno relief\n(θ = 0)", "drew\n'wait'", "drew\n'go'"], fontsize=7.5)
        ax.text(2.3, -3.6, "random verdict with relief (θ = 0.9)", ha="center", va="top", fontsize=7.5, color=INK,
                transform=ax.transData, clip_on=False)
        ax.plot([1.72, 2.88], [-3.3, -3.3], color=INK2, lw=0.8, clip_on=False)
        ax.set_xlim(-0.55, 3.6); ax.set_ylim(0, 9)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of decisions: did not act, state was good")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_two_sided.png")); plt.close(fig)
    for P in (iv.ANXIOUS, iv.FEARFUL):
        none = out[(P.label, "none")]; a9, g9, w9 = out[(P.label, 0.9)]
        print(f"fig3 {P.label}: missed none {100*none.missed:.1f}% | θ=.9 wait {100*w9.missed:.1f}% go {100*g9.missed:.1f}% | calibrated {100*out[('calibrated','none')].missed:.1f}%")


# ================================================================ Table 1: what the whole procedure buys
def compressed_interval(P, c, seed=31):
    """Everyone must act or give up after one observation (no waiting allowed)."""
    th, X, vd, suc = iv._streams(seed, N)
    x = th + iv.SIGMA * X[:, 0]
    L = 2 * x / iv.SIGMA ** 2 * P.gamma
    ps = iv.p_success(iv.sigmoid(L))
    act = ps * iv.R_OK + (1 - ps) * (iv.R_BAD - P.r_blame) > 0
    pay = np.where(act, np.where(suc, iv.R_OK, iv.R_BAD), 0.0) + c
    return dict(payoff=float(pay.mean()), avoidable=float((act & (th < 0)).mean()),
                missed=float(((~act) & (th > 0)).mean()), wrong=float((act & ~suc).mean()), steps=1.0)


def _episode_arrays(P, device, c, p):
    """Per-episode outcome arrays for one person under one device, seed 31, N episodes
    (common random numbers across devices and persons)."""
    th, X, vd, suc = iv._streams(31, N)
    ones = np.ones(N, bool)
    up0, lo0 = iv.thresholds_for(P)
    if device == "own":
        d = iv._run_core(th, X, suc, iv.SIGMA, P, up0, lo0, up0, lo0, ones)
    elif device == "interval only":
        d = iv._run_core(th, X, suc, iv.SIGMA, P, up0, lo0, up0, lo0, ones, k_wait=3)
    elif device in ("procedure", "relief only"):
        go = vd.random(N) < p
        ug, lg = iv.thresholds_for(P, relief_go=0.9)
        uw, lw = iv.thresholds_for(P, relief_wait=0.9)
        d = iv._run_core(th, X, suc, iv.SIGMA, P, ug, lg, uw, lw, go, k_wait=(3 if device == "procedure" else 0))
    elif device == "agent-timed prompt":
        agent = iv.Person(gamma=2.0, label="agent")
        up_a, lo_a = iv.thresholds_for(agent)
        L = np.zeros(N); done = np.zeros(N, bool); acted = np.zeros(N, bool); steps = np.zeros(N, int)
        for t in range(1, iv.H + 1):
            live = ~done
            x = th + iv.SIGMA * X[:, t - 1]
            L = np.where(live, L + 2.0 * x / iv.SIGMA ** 2, L)
            steps += live
            la = 2.0 * L
            stop = live & ((la >= up_a[t]) | (la <= lo_a[t]) | (t == iv.H))
            lp = P.gamma * L
            ps = iv.p_success(iv.sigmoid(lp))
            act = ps * iv.R_OK + (1 - ps) * (iv.R_BAD - P.r_blame) > 0
            acted |= stop & act
            done |= stop
        d = dict(pay=np.where(acted, np.where(suc, iv.R_OK, iv.R_BAD), 0.0) + iv.C * steps,
                 avoidable=acted & (th < 0), missed=(~acted) & (th > 0))
    elif device == "compressed interval":
        x = th + iv.SIGMA * X[:, 0]
        Lp = 2 * x / iv.SIGMA ** 2 * P.gamma
        ps = iv.p_success(iv.sigmoid(Lp))
        act = ps * iv.R_OK + (1 - ps) * (iv.R_BAD - P.r_blame) > 0
        d = dict(pay=np.where(act, np.where(suc, iv.R_OK, iv.R_BAD), 0.0) + c, avoidable=act & (th < 0), missed=(~act) & (th > 0))
    return dict(pay=d["pay"].astype(float), avoidable=d["avoidable"].astype(float), missed=d["missed"].astype(float))


def table1_paired(p=0.25, suffix=""):
    """Paired statistics for Table 2 (main text): 95% half-widths of the population cells
    computed from the per-episode population average (the four persons share random
    streams, so they are not independent samples), and paired differences against 'own
    judgement' per person and for the population.  Writes table1_paired{suffix}.csv."""
    rows = []
    devices = ("own", "procedure", "interval only", "relief only", "agent-timed prompt", "compressed interval")
    for c in (-1.0, -0.5):
        old_c = iv.C; iv.C = c; iv.dp_thresholds.cache_clear()
        arrs = {(P.label, dev): _episode_arrays(P, dev, c, p) for P in iv.PEOPLE for dev in devices}
        for dev in devices:
            pop = {k: np.mean([arrs[(P.label, dev)][k] for P in iv.PEOPLE], axis=0) for k in ("pay", "avoidable", "missed")}
            base = {k: np.mean([arrs[(P.label, "own")][k] for P in iv.PEOPLE], axis=0) for k in ("pay", "avoidable", "missed")}
            row = dict(step_cost=c, person="population (equal mix)", device=dev)
            for k in ("pay", "avoidable", "missed"):
                row[f"{k}_mean"] = float(pop[k].mean())
                row[f"{k}_halfwidth95"] = float(1.96 * pop[k].std(ddof=1) / np.sqrt(N))
                diff = pop[k] - base[k]
                row[f"{k}_diff_vs_own"] = float(diff.mean())
                row[f"{k}_diff_halfwidth95"] = float(1.96 * diff.std(ddof=1) / np.sqrt(N))
            rows.append(row)
            for P in iv.PEOPLE:
                a, b = arrs[(P.label, dev)], arrs[(P.label, "own")]
                row = dict(step_cost=c, person=P.label, device=dev)
                for k in ("pay", "avoidable", "missed"):
                    diff = a[k] - b[k]
                    row[f"{k}_mean"] = float(a[k].mean())
                    row[f"{k}_halfwidth95"] = float(1.96 * a[k].std(ddof=1) / np.sqrt(N))
                    row[f"{k}_diff_vs_own"] = float(diff.mean())
                    row[f"{k}_diff_halfwidth95"] = float(1.96 * diff.std(ddof=1) / np.sqrt(N))
                rows.append(row)
        iv.C = old_c; iv.dp_thresholds.cache_clear()
    write_csv(os.path.join(RES, f"table1_paired{suffix}.csv"), rows)
    for r in rows:
        if r["person"].startswith("population") and r["step_cost"] == -0.5:
            print(f"  paired C=-0.5 {r['device']:>19}: payoff {r['pay_mean']:+.3f} ±{r['pay_halfwidth95']:.3f}  diff vs own {r['pay_diff_vs_own']:+.3f} ±{r['pay_diff_halfwidth95']:.3f} | avoidable ±{100*r['avoidable_halfwidth95']:.3f}pp missed ±{100*r['missed_halfwidth95']:.3f}pp")


def table1(p=0.25, suffix=""):
    """Population of the four kinds of person in equal shares.  p = probability that the
    verdict says "go" (main text 0.25: one line in four changes under either casting
    method; p = 0.5 reported in the Supplement as table1_p05.csv).  Columns: act at once
    (no evidence), never act, own judgement, the whole procedure (interval k=2 +
    random verdict theta=0.9), and its two parts.  Two stakes ratios: a step of
    waiting = 1% and 0.5% of the catastrophe."""
    rows = []
    for c in (-1.0, -0.5):
        old_c = iv.C; iv.C = c; iv.dp_thresholds.cache_clear()
        cells = {}
        for P in iv.PEOPLE:
            cells[(P.label, "own")] = iv.simulate(P, "none", n=N, seed=31)
            cells[(P.label, "procedure")] = iv.simulate(P, "oracle", n=N, seed=31, k=2, theta=0.9, p=p)
            # forced_wait's k is the first step at which a decision is allowed, so k=3 is
            # two imposed steps beyond the first observation: the same interval as the
            # procedure's oracle(k=2), which uses k_wait = k + 1 (see interval.simulate).
            cells[(P.label, "interval only")] = iv.simulate(P, "forced_wait", n=N, seed=31, k=3)
            cells[(P.label, "relief only")] = iv.simulate(P, "verdict", n=N, seed=31, theta=0.9, p=p)
        act_now_pay = 0.5 * (iv.Q * iv.R_OK + (1 - iv.Q) * iv.R_BAD) + 0.5 * ((1 - iv.Q) * iv.R_OK + iv.Q * iv.R_BAD) + c
        for col in ("own", "procedure", "interval only", "relief only"):
            for P in iv.PEOPLE:
                r = cells[(P.label, col)]
                rows.append(dict(step_cost=c, person=P.label, device=col, payoff=r.payoff, avoidable=r.avoidable,
                                 missed=r.missed, wrong=r.wrong, steps=r.steps))
            rows.append(dict(step_cost=c, person="population (equal mix)", device=col,
                             payoff=np.mean([cells[(P.label, col)].payoff for P in iv.PEOPLE]),
                             avoidable=np.mean([cells[(P.label, col)].avoidable for P in iv.PEOPLE]),
                             missed=np.mean([cells[(P.label, col)].missed for P in iv.PEOPLE]),
                             wrong=np.mean([cells[(P.label, col)].wrong for P in iv.PEOPLE]),
                             steps=np.mean([cells[(P.label, col)].steps for P in iv.PEOPLE])))
        rows.append(dict(step_cost=c, person="population (equal mix)", device="act at once", payoff=act_now_pay,
                         avoidable=0.5, missed=0.0, wrong=0.5 * (1 - iv.Q) + 0.5 * iv.Q, steps=1))
        rows.append(dict(step_cost=c, person="population (equal mix)", device="never act", payoff=0.0,
                         avoidable=0.0, missed=0.5, wrong=0.0, steps=0))
        iv.C = old_c; iv.dp_thresholds.cache_clear()
    # compressed interval: decide after a single observation (the shape of an approval prompt)
    for c in (-1.0, -0.5):
        res = [compressed_interval(P, c) for P in iv.PEOPLE]
        for P, r in zip(iv.PEOPLE, res):
            rows.append(dict(step_cost=c, person=P.label, device="compressed interval", **r))
        rows.append(dict(step_cost=c, person="population (equal mix)", device="compressed interval",
                         **{k: float(np.mean([r[k] for r in res])) for k in res[0]}))
    # agent-timed prompt: an agent that reads each observation twice as strongly (gamma 2)
    # stops when its own thresholds are crossed; the person must then act or give up.
    for c in (-1.0, -0.5):
        old_c = iv.C; iv.C = c; iv.dp_thresholds.cache_clear()
        res = [iv.simulate_prompt(P, agent_gamma=2.0, n=N, seed=31) for P in iv.PEOPLE]
        for P, r in zip(iv.PEOPLE, res):
            rows.append(dict(step_cost=c, person=P.label, device="agent-timed prompt", payoff=r.payoff,
                             avoidable=r.avoidable, missed=r.missed, wrong=r.wrong, steps=r.steps))
        rows.append(dict(step_cost=c, person="population (equal mix)", device="agent-timed prompt",
                         payoff=float(np.mean([r.payoff for r in res])), avoidable=float(np.mean([r.avoidable for r in res])),
                         missed=float(np.mean([r.missed for r in res])), wrong=float(np.mean([r.wrong for r in res])),
                         steps=float(np.mean([r.steps for r in res]))))
        iv.C = old_c; iv.dp_thresholds.cache_clear()
    write_csv(os.path.join(RES, f"table1{suffix}.csv"), rows)
    table1_paired(p, suffix)
    print(f"table1 p={p} (population, equal mix):")
    for c in (-1.0, -0.5):
        for col in ("act at once", "never act", "own", "procedure", "interval only", "relief only", "agent-timed prompt", "compressed interval"):
            r = [x for x in rows if x["step_cost"] == c and x["person"].startswith("population") and x["device"] == col][0]
            print(f"  C={c:+.1f} {col:>14}: payoff {r['payoff']:+6.2f}  avoidable {100*r['avoidable']:5.2f}%  missed {100*r['missed']:5.1f}%")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); table1(); table1(p=0.5, suffix="_p05")
    print("done ->", FIGS)
