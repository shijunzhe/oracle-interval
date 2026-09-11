"""Precision checks for Supplementary Note 1: the reported numbers under a finer evidence
grid and more or fewer quadrature nodes.  Writes results/convergence.csv."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import interval as iv

N = 100_000
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "convergence.csv")
CONFIGS = [("baseline: 2401 points, 31 nodes", 2401, 31), ("4801 points, 31 nodes", 4801, 31),
           ("9601 points, 31 nodes", 9601, 31), ("2401 points, 15 nodes", 2401, 15), ("2401 points, 61 nodes", 2401, 61)]


def set_numerics(npts, nodes):
    iv._GRID = np.linspace(-30.0, 30.0, npts)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    iv._GH_X, iv._GH_W = x, w / w.sum()
    iv.dp_thresholds.cache_clear()


rows = []
for label, npts, nodes in CONFIGS:
    set_numerics(npts, nodes)
    for P in iv.PEOPLE:
        own = iv.simulate(P, "none", n=N, seed=31)
        proc = iv.simulate(P, "oracle", n=N, seed=31, k=2, theta=iv.THETA, p=0.25)
        rows.append(dict(config=label, person=P.label, own_payoff=own.payoff, own_avoidable=own.avoidable, own_missed=own.missed,
                         procedure_payoff=proc.payoff, procedure_avoidable=proc.avoidable, procedure_missed=proc.missed))
    for k in (0, 2, 3):
        r = iv.simulate(iv.OVERCONFIDENT, "forced_wait" if k else "none", n=N, seed=11, k=k + 1)
        rows.append(dict(config=label, person=f"overconfident, Fig. 2 k={k}", own_payoff=r.payoff, own_avoidable=r.avoidable,
                         own_missed=r.missed, procedure_payoff="", procedure_avoidable="", procedure_missed=""))
    h = iv.simulate_handoff(iv.OVERCONFIDENT, 4.0, "required", k_required=2, n=N, seed=31)
    rows.append(dict(config=label, person="overconfident, Table 4 C (gamma_a = 4)", own_payoff=h.payoff, own_avoidable=h.avoidable,
                     own_missed=h.missed, procedure_payoff="", procedure_avoidable="", procedure_missed=""))
    print(label, "done")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
base = [r for r in rows if r["config"].startswith("baseline")]
for r in rows:
    b = [x for x in base if x["person"] == r["person"]][0]
    diffs = [abs(float(r[k]) - float(b[k])) for k in r if k not in ("config", "person") and r[k] != ""]
    print(f"{r['config']:32s} {r['person']:40s} max |diff| {max(diffs):.5f}")
