"""make_supp.py — supplementary simulations (M3): S-A stakes ratio, S-B verdict probability,
S-W subjective relief (by construction), S-D theta curve, S-E self-imposed rule, S-F pressure
strength, S-G verdict read as evidence, S-H mirror verdict, S-M moral hazard, robustness
table.   python3 code/make_supp.py [n]
"""
# NOTE (2026-09-06): the figures this script draws are superseded by code/plot_supp.py, which redraws them
# from results/*.csv under the figure rules in PLAN §8 (no overlaps, legends outside data, lowercase bold
# panel labels, 300 dpi + PDF). Run this script for the numbers, then `python3 code/plot_supp.py` for the figures.
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import interval as iv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
BASE = dict(R_OK=iv.R_OK, R_BAD=iv.R_BAD, C=iv.C, Q=iv.Q, SIGMA=iv.SIGMA, H=iv.H)


def setp(**kw):
    for k, v in {**BASE, **kw}.items():
        setattr(iv, k, v)
    iv.dp_thresholds.cache_clear()


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def pop_mean(results, key):
    return float(np.mean([getattr(r, key) for r in results]))


# ---------------------------------------------------------------- S-A stakes ratio
def s_a():
    rows = []
    grid = [(20, -100, 0.95, c) for c in (-1.0, -0.5, -0.25)] + [(20, -300, 0.99, -1.0), (50, -1000, 0.99, -1.0), (50, -1000, 0.99, -0.25)]
    for rok, rbad, q, c in grid:
        setp(R_OK=rok, R_BAD=rbad, Q=q, C=c)
        own = [iv.simulate(P, "none", n=N, seed=41, sigma=iv.SIGMA) for P in iv.PEOPLE]
        proc = [iv.simulate(P, "oracle", n=N, seed=41, sigma=iv.SIGMA, k=2, theta=iv.THETA) for P in iv.PEOPLE]
        ratio = abs(c) / abs(rbad)
        rows.append(dict(R_OK=rok, R_BAD=rbad, Q=q, C=c, step_over_catastrophe=ratio,
                         own_payoff=pop_mean(own, "payoff"), proc_payoff=pop_mean(proc, "payoff"),
                         d_payoff=pop_mean(proc, "payoff") - pop_mean(own, "payoff"),
                         own_avoidable=pop_mean(own, "avoidable"), proc_avoidable=pop_mean(proc, "avoidable"),
                         own_missed=pop_mean(own, "missed"), proc_missed=pop_mean(proc, "missed"),
                         overconf_d_payoff=proc[1].payoff - own[1].payoff, calibrated_d_payoff=proc[0].payoff - own[0].payoff))
    setp()
    write_csv(os.path.join(RES, "sA_stakes.csv"), rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
    x = [100 * r["step_over_catastrophe"] for r in rows]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(x, [r["d_payoff"] for r in rows], color=INK, s=28, label="population (equal mix)", zorder=3)
    ax.scatter(x, [r["overconf_d_payoff"] for r in rows], color=ORANGE, s=28, marker="s", label="overconfident person", zorder=3)
    ax.scatter(x, [r["calibrated_d_payoff"] for r in rows], color=BLUE, s=28, marker="^", label="calibrated person", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("cost of one step of waiting, as % of the catastrophe (log scale)")
    ax.set_ylabel("payoff: whole procedure − own judgement")
    ax.legend(frameon=False, fontsize=7.5); ax.set_title("S-A  the interval pays when the stakes are high", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "supp_sA.png")); plt.close(fig)
    for r in rows:
        print(f"S-A {r['R_OK']:>3}/{r['R_BAD']:>5} C={r['C']:+.2f} step/cat={100*r['step_over_catastrophe']:.2f}%  Δpayoff pop {r['d_payoff']:+.2f} overconf {r['overconf_d_payoff']:+.2f} calib {r['calibrated_d_payoff']:+.2f} | avoidable {100*r['own_avoidable']:.2f}→{100*r['proc_avoidable']:.2f} missed {100*r['own_missed']:.1f}→{100*r['proc_missed']:.1f}")


# ---------------------------------------------------------------- S-B verdict probability (linear)
def s_b():
    rows = []
    ps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for p in ps:
        res = {P.label: iv.simulate(P, "verdict", n=N, seed=43, p=p, theta=iv.THETA) for P in iv.PEOPLE}
        rows.append(dict(p=p, **{f"payoff_{k}": v.payoff for k, v in res.items()}, **{f"missed_{k}": v.missed for k, v in res.items()},
                         pop_payoff=np.mean([v.payoff for v in res.values()]), pop_missed=np.mean([v.missed for v in res.values()])))
    write_csv(os.path.join(RES, "sB_p.csv"), rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
    ax.plot(ps, [100 * r["missed_anxious"] for r in rows], color=BLUE, marker="o", lw=2, label="anxious")
    ax.plot(ps, [100 * r["missed_fearful"] for r in rows], color=ORANGE, marker="o", lw=2, label="fearful")
    ax.plot(ps, [100 * r["pop_missed"] for r in rows], color=INK, marker="o", lw=2, ls="--", label="population (equal mix)")
    ax.set_xlabel("probability that the verdict says 'go'  (0 = always 'wait', 1 = always 'go')")
    ax.set_ylabel("% missed opportunities"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("S-B  the draw itself buys nothing in expectation: outcomes are linear in p", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "supp_sB.png")); plt.close(fig)
    print("S-B pop payoff by p:", [f"{r['pop_payoff']:+.2f}" for r in rows], " pop missed:", [f"{100*r['pop_missed']:.1f}" for r in rows])


# ---------------------------------------------------------------- S-W subjective relief (by construction)
def s_w():
    rows = []
    for P in (iv.ANXIOUS, iv.FEARFUL):
        none = iv.simulate(P, "none", n=N, seed=45)
        _, go, wait = iv.simulate(P, "verdict", n=N, seed=45, theta=iv.THETA, split=True)
        for lbl, r, relief in (("none", none, 0.0), ("drew wait (θ=relief)", wait, iv.THETA if P is iv.ANXIOUS else 0.0),
                               ("drew go (θ=relief)", go, iv.THETA if P is iv.FEARFUL else 0.0)):
            anx = P.c_anx * (1 - relief) * r.steps
            blame = P.r_blame * (1 - relief) * r.wrong
            rows.append(dict(person=P.label, condition=lbl, steps=r.steps, anxiety_endured=anx, expected_blame=blame,
                             missed=r.missed, payoff=r.payoff))
    write_csv(os.path.join(RES, "sW_subjective.csv"), rows)
    for r in rows:
        print(f"S-W {r['person']:>8} {r['condition']:>18}: anxiety endured {r['anxiety_endured']:.2f}  expected blame {r['expected_blame']:.2f}  (by construction)")


# ---------------------------------------------------------------- S-D theta curve
def s_d():
    rows = []
    ths = [0.0, 0.25, 0.5, 0.7, 0.9, 1.0]
    for th in ths:
        _, _, an_wait = iv.simulate(iv.ANXIOUS, "verdict", n=N, seed=47, theta=th, split=True)
        _, fe_go, _ = iv.simulate(iv.FEARFUL, "verdict", n=N, seed=47, theta=th, split=True)
        rows.append(dict(theta=th, anxious_wait_missed=an_wait.missed, fearful_go_missed=fe_go.missed,
                         anxious_wait_payoff=an_wait.payoff, fearful_go_payoff=fe_go.payoff))
    cal = iv.simulate(iv.CALIBRATED, "none", n=N, seed=47)
    write_csv(os.path.join(RES, "sD_theta.csv"), rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
    ax.plot(ths, [100 * r["anxious_wait_missed"] for r in rows], color=BLUE, marker="o", lw=2, label="anxious person who drew 'wait'")
    ax.plot(ths, [100 * r["fearful_go_missed"] for r in rows], color=ORANGE, marker="o", lw=2, label="fearful person who drew 'go'")
    ax.axhline(100 * cal.missed, color=INK2, ls="--", lw=1); ax.text(0.02, 100 * cal.missed + 0.15, "calibrated person", fontsize=7.5, color=INK2)
    ax.set_xlabel("θ — share of the pressure the verdict lifts"); ax.set_ylabel("% missed opportunities")
    ax.legend(frameon=False, fontsize=7.5); ax.set_title("S-D  correction grows with relief; full relief = calibrated", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "supp_sD.png")); plt.close(fig)
    print("S-D anxious/wait missed:", [f"{100*r['anxious_wait_missed']:.1f}" for r in rows], " fearful/go:", [f"{100*r['fearful_go_missed']:.1f}" for r in rows])


# ---------------------------------------------------------------- S-E self-ignorance
def s_e():
    """A self-imposed rule: the person discounts their own certainty by the factor f, i.e.
    both of their own thresholds are multiplied by f (perceived units).  f_star is the
    factor that restores the calibrated person's action threshold in true-evidence units,
    gamma*up_cal/up_own: the number a person would need to know their own gamma to set."""
    rows = []
    cal = iv.simulate(iv.CALIBRATED, "none", n=N, seed=49)
    up_cal, _ = iv.thresholds_for(iv.CALIBRATED)
    for g in (1.0, 2.0, 4.0, 6.0):
        P = iv.Person(gamma=g)
        up_own, _ = iv.thresholds_for(P)
        f_star = float(g * up_cal[1] / up_own[1])
        own = iv.simulate(P, "none", n=N, seed=49)
        wait2 = iv.simulate(P, "forced_wait", n=N, seed=49, k=3)
        row = dict(gamma=g, f_star=f_star, own_avoidable=own.avoidable, own_missed=own.missed, own_payoff=own.payoff,
                   forced_wait2_avoidable=wait2.avoidable, forced_wait2_missed=wait2.missed, forced_wait2_payoff=wait2.payoff)
        for f in (1.25, 1.5, 2.0):
            r = iv.simulate(P, "self_rule", n=N, seed=49, f=f)
            row[f"f{f}_avoidable"] = r.avoidable; row[f"f{f}_missed"] = r.missed; row[f"f{f}_payoff"] = r.payoff
        r = iv.simulate(P, "self_rule", n=N, seed=49, f=f_star)
        row["fstar_avoidable"] = r.avoidable; row["fstar_missed"] = r.missed; row["fstar_payoff"] = r.payoff
        rows.append(row)
        print(f"S-E γ={g}: f*={f_star:.2f} | own {100*own.avoidable:.2f}%/{100*own.missed:.1f}%/{own.payoff:.2f} | wait2 {100*wait2.avoidable:.2f}%/{100*wait2.missed:.1f}%/{wait2.payoff:.2f} | f=1.25 {100*row['f1.25_avoidable']:.2f}%/{100*row['f1.25_missed']:.1f}%/{row['f1.25_payoff']:.2f} | f* {100*r.avoidable:.2f}%/{100*r.missed:.1f}%/{r.payoff:.2f}  (calibrated {100*cal.avoidable:.2f}%/{100*cal.missed:.1f}%/{cal.payoff:.2f})")
    write_csv(os.path.join(RES, "sE_selfrule.csv"), rows)


# ---------------------------------------------------------------- S-G the verdict read as evidence
def s_g():
    """The consulter also treats the verdict as evidence about the world: perceived
    log-odds shift by +delta ('go') or -delta ('wait').  Random verdict, theta=iv.THETA, p=0.25.
    Split by the verdict drawn; the same person alone is summarised on the same episode
    subsets (common random numbers), so go_payoff - go_own_payoff is a paired difference."""
    rows = []
    for P in iv.PEOPLE:
        th, X, vd, suc = iv._streams(23, N)
        go_mask = vd.random(N) < 0.25            # the same draw simulate() makes for seed 23
        up0, lo0 = iv.thresholds_for(P)
        own_d = iv._run_core(th, X, suc, iv.SIGMA, P, up0, lo0, up0, lo0, np.ones(N, bool))
        own, own_go, own_wt = iv._summarise(own_d), iv._summarise(own_d, go_mask), iv._summarise(own_d, ~go_mask)
        for d in (0.0, 0.5, 1.0, 1.5, 2.0):
            r, rg, rw = iv.simulate(P, "verdict", n=N, seed=23, theta=iv.THETA, p=0.25, delta=d, split=True)
            assert rg.n == own_go.n
            rows.append(dict(person=P.label, delta=d, own_payoff=own.payoff, own_avoidable=own.avoidable, own_missed=own.missed,
                             payoff=r.payoff, avoidable=r.avoidable, missed=r.missed,
                             go_payoff=rg.payoff, go_avoidable=rg.avoidable, go_missed=rg.missed, go_own_payoff=own_go.payoff,
                             wait_payoff=rw.payoff, wait_avoidable=rw.avoidable, wait_missed=rw.missed, wait_own_payoff=own_wt.payoff))
        print(f"S-G {P.label:>13}: own {own.payoff:.2f} | " + " ".join(f"δ={x['delta']}: all {x['payoff']-x['own_payoff']:+.2f} go {x['go_payoff']-x['go_own_payoff']:+.2f} wait {x['wait_payoff']-x['wait_own_payoff']:+.2f}" for x in rows if x['person'] == P.label))
    write_csv(os.path.join(RES, "sG_evidence.csv"), rows)


# ---------------------------------------------------------------- S-H a verdict that predicts the person
def s_h():
    """Mirror verdict (says what the person's own leaning after k=2 observations says)
    against two random verdicts drawn at the same step: one at p=0.25 as in the main text,
    one at p equal to the mirror's own share of 'go' verdicts (the matched control), all
    with relief theta=iv.THETA and the evidence channel delta in {0, 1}."""
    rows = []
    for P in iv.PEOPLE:
        for d in (0.0, 1.0):
            mir, mgo, _ = iv.simulate(P, "mirror", n=N, seed=23, k=2, theta=iv.THETA, delta=d, split=True)
            share = mgo.n / N
            rnd = iv.simulate(P, "oracle", n=N, seed=23, k=2, theta=iv.THETA, p=0.25, delta=d)
            rnm = iv.simulate(P, "oracle", n=N, seed=23, k=2, theta=iv.THETA, p=share, delta=d)
            rows.append(dict(person=P.label, delta=d, mirror_go_share=share,
                             random_p025_payoff=rnd.payoff, random_p025_avoidable=rnd.avoidable, random_p025_missed=rnd.missed,
                             random_matched_payoff=rnm.payoff, random_matched_avoidable=rnm.avoidable, random_matched_missed=rnm.missed,
                             mirror_payoff=mir.payoff, mirror_avoidable=mir.avoidable, mirror_missed=mir.missed))
            print(f"S-H {P.label:>13} δ={d}: go-share {share:.3f} | random p=.25 {rnd.payoff:.2f}/{100*rnd.avoidable:.2f}% | matched {rnm.payoff:.2f}/{100*rnm.avoidable:.2f}% | mirror {mir.payoff:.2f}/{100*mir.avoidable:.2f}%  ratio vs matched {mir.avoidable/rnm.avoidable:.2f}")
    write_csv(os.path.join(RES, "sH_mirror.csv"), rows)


# ---------------------------------------------------------------- S-M moral hazard: two faults in one person
def s_m():
    """gamma in {1,GAMMA_OVER} x R_blame in {0,R_BLAME} under the random verdict theta=iv.THETA, split by verdict."""
    rows = []
    for g in (1.0, iv.GAMMA_OVER):
        for rb in (0.0, iv.R_BLAME):
            P = iv.Person(gamma=g, r_blame=rb, label=f"gamma{g:g}_blame{rb:g}")
            own = iv.simulate(P, "none", n=N, seed=23)
            _, go, wt = iv.simulate(P, "verdict", n=N, seed=23, theta=iv.THETA, p=0.25, split=True)
            rows.append(dict(gamma=g, r_blame=rb, own_payoff=own.payoff, own_avoidable=own.avoidable, own_missed=own.missed,
                             go_payoff=go.payoff, go_avoidable=go.avoidable, go_missed=go.missed,
                             wait_payoff=wt.payoff, wait_avoidable=wt.avoidable, wait_missed=wt.missed))
            print(f"S-M γ={g:g} R_blame={rb:g}: own {own.payoff:.2f}/{100*own.avoidable:.2f}%/{100*own.missed:.1f}% | drew go {go.payoff:.2f}/{100*go.avoidable:.2f}%/{100*go.missed:.1f}% | drew wait {wt.payoff:.2f}/{100*wt.avoidable:.2f}%/{100*wt.missed:.1f}%")
    write_csv(os.path.join(RES, "sM_moral_hazard.csv"), rows)


# ---------------------------------------------------------------- S-F pressure strength
def s_f():
    rows = []
    cal = iv.simulate(iv.CALIBRATED, "none", n=N, seed=51)
    for c_anx in (0.5, 1.0, 2.0, 4.0):
        P = iv.Person(c_anx=c_anx, label="anxious")
        none = iv.simulate(P, "none", n=N, seed=51)
        _, _, wait = iv.simulate(P, "verdict", n=N, seed=51, theta=iv.THETA, split=True)
        rows.append(dict(person="anxious", strength=c_anx, none_missed=none.missed, relieved_missed=wait.missed, calibrated_missed=cal.missed))
    for rb in (50, 100, 200, 400):
        P = iv.Person(r_blame=rb, label="fearful")
        none = iv.simulate(P, "none", n=N, seed=51)
        _, go, _ = iv.simulate(P, "verdict", n=N, seed=51, theta=iv.THETA, split=True)
        rows.append(dict(person="fearful", strength=rb, none_missed=none.missed, relieved_missed=go.missed, calibrated_missed=cal.missed))
    write_csv(os.path.join(RES, "sF_pressure.csv"), rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=200, sharey=True)
    for ax, person, col, xlabel in ((axes[0], "anxious", BLUE, "extra cost of each step of waiting, C_anx"),
                                    (axes[1], "fearful", ORANGE, "extra cost of blame after a failure, R_blame")):
        sub = [r for r in rows if r["person"] == person]
        xs = [r["strength"] for r in sub]
        ax.plot(xs, [100 * r["none_missed"] for r in sub], color=col, marker="o", lw=2, label=f"{person}, alone")
        ax.plot(xs, [100 * r["relieved_missed"] for r in sub], color=col, marker="o", lw=2, ls="--", label=f"{person}, drew the relieving verdict (θ=.9)")
        ax.axhline(100 * cal.missed, color=INK2, ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.minorticks_off()
        ax.set_xlabel(xlabel); ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("% missed opportunities")
    axes[0].set_title("S-F  the stronger the pressure, the larger the correction", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "supp_sF.png")); plt.close(fig)
    for r in rows:
        print(f"S-F {r['person']:>8} strength {r['strength']:>6}: missed alone {100*r['none_missed']:.1f}% → relieved {100*r['relieved_missed']:.1f}%  (calibrated {100*r['calibrated_missed']:.1f}%)")


# ---------------------------------------------------------------- S-N: two controls (review, 2026-09-08)
def s_n():
    """S-N: two controls asked for in review.  (i) 'blind wait': the two imposed steps of
    the main text, but no observation arrives during them, so the wait carries time and
    cost and no evidence.  (ii) 'protection both ways, no draw': the relief of theta = 0.9
    applied to both extra costs at once, for everyone, with no verdict, which is what the
    second rule of Section 8 asks an institution to do; with and without the interval.
    Same seed as Table 2's supplementary companions."""
    th = iv.THETA
    rows = []
    for P in iv.PEOPLE:
        both = iv.Person(gamma=P.gamma, c_anx=P.c_anx * (1 - th), r_blame=P.r_blame * (1 - th), label=P.label)
        cells = [
            ("own judgement", iv.simulate(P, "none", n=N, seed=23)),
            ("interval only", iv.simulate(P, "forced_wait", n=N, seed=23, k=3)),
            ("blind wait", iv.simulate(P, "blind_wait", n=N, seed=23, k=3)),
            ("protection both ways, no draw", iv.simulate(both, "none", n=N, seed=23)),
            ("protection both ways + interval", iv.simulate(both, "forced_wait", n=N, seed=23, k=3)),
            ("the procedure", iv.simulate(P, "oracle", n=N, seed=23, k=2, p=0.25, theta=th)),
        ]
        for name, r in cells:
            rows.append(dict(person=P.label, setting=name, payoff=r.payoff, avoidable=r.avoidable,
                             missed=r.missed, steps=r.steps))
    write_csv(os.path.join(RES, "sN_controls.csv"), rows)
    for r in rows:
        print(f"S-N {r['person']:>13} {r['setting']:>32}: payoff {r['payoff']:5.2f} avoidable {100*r['avoidable']:.2f}% missed {100*r['missed']:.1f}% steps {r['steps']:.2f}")


# ---------------------------------------------------------------- S-P: how overconfident the agent is (Table S9)
def s_p():
    """Agent-timed decision with the agent's gamma in {1, 2, 4, 6}, four persons (seed 31 as Table 3)."""
    rows = []
    for ga in (1.0, 2.0, 4.0, 6.0):
        for P in iv.PEOPLE:
            own = iv.simulate(P, "none", n=N, seed=31)
            r = iv.simulate_prompt(P, agent_gamma=ga, n=N, seed=31)
            rows.append(dict(agent_gamma=ga, person=P.label, own_payoff=own.payoff, payoff=r.payoff, avoidable=r.avoidable, missed=r.missed))
            print(f"S-P agent γ={ga:g} {P.label:13s}: own {own.payoff:.2f} -> agent-timed {r.payoff:.2f}  avoidable {100*r.avoidable:.2f}%  missed {100*r.missed:.1f}%")
    write_csv(os.path.join(RES, "sP_agent.csv"), rows)


# ---------------------------------------------------------------- robustness table
def robustness():
    rows = []
    def cell(tag, **kw):
        setp(**kw)
        sig = kw.get("SIGMA", BASE["SIGMA"])
        out = {}
        for P in iv.PEOPLE:
            out[P.label] = iv.simulate(P, "none", n=N, seed=53, sigma=sig)
        ov2 = iv.simulate(iv.OVERCONFIDENT, "forced_wait", n=N, seed=53, sigma=sig, k=3)
        _, _, an_w = iv.simulate(iv.ANXIOUS, "verdict", n=N, seed=53, sigma=sig, theta=iv.THETA, split=True)
        _, fe_g, _ = iv.simulate(iv.FEARFUL, "verdict", n=N, seed=53, sigma=sig, theta=iv.THETA, split=True)
        la = iv.simulate_lookahead(iv.CALIBRATED, n=N, seed=53, sigma=sig)
        la2 = iv.simulate_lookahead(iv.CALIBRATED, n=N, seed=53, sigma=sig, k_wait=3)
        lao = iv.simulate_lookahead(iv.OVERCONFIDENT, n=N, seed=53, sigma=sig)
        lao2 = iv.simulate_lookahead(iv.OVERCONFIDENT, n=N, seed=53, sigma=sig, k_wait=3)
        rows.append(dict(variant=tag, cal_payoff=out["calibrated"].payoff, lookahead_payoff=la.payoff,
                         lookahead_wait2_payoff=la2.payoff, overconf_lookahead_payoff=lao.payoff,
                         overconf_lookahead_wait2_payoff=lao2.payoff, overconf_payoff=out["overconfident"].payoff,
                         overconf_wait2_payoff=ov2.payoff,
                         overconf_avoidable=out["overconfident"].avoidable, overconf_avoidable_wait2=ov2.avoidable,
                         anx_missed=out["anxious"].missed, anx_missed_wait_relief=an_w.missed,
                         fear_missed=out["fearful"].missed, fear_missed_go_relief=fe_g.missed, cal_missed=out["calibrated"].missed))
        setp()
    cell("main text")
    cell("sigma=1.0", SIGMA=1.0); cell("sigma=1.5", SIGMA=1.5)
    cell("H=8", H=8); cell("H=40", H=40)
    cell("Q=0.90", Q=0.90); cell("Q=0.99", Q=0.99); cell("Q=1.0 (no luck)", Q=1.0)
    write_csv(os.path.join(RES, "robustness.csv"), rows)
    for r in rows:
        print(f"ROB {r['variant']:>16}: DP {r['cal_payoff']:+.2f} vs lookahead {r['lookahead_payoff']:+.2f} | overconf avoidable {100*r['overconf_avoidable']:.2f}→wait2 {100*r['overconf_avoidable_wait2']:.2f} | anx missed {100*r['anx_missed']:.1f}→{100*r['anx_missed_wait_relief']:.1f} | fear {100*r['fear_missed']:.1f}→{100*r['fear_missed_go_relief']:.1f} | cal {100*r['cal_missed']:.1f}")


if __name__ == "__main__":
    s_a(); s_b(); s_w(); s_d(); s_e(); s_f(); s_g(); s_h(); s_m(); s_n(); s_p(); robustness()
    print("done")
