"""interval.py — GATE simulation baseline (T1.1, design v3.2).

One model, four kinds of person, two kinds of device.

World.  A binary latent state theta in {+1,-1} (equiprobable).  Each step brings
one noisy observation x ~ N(theta, sigma^2).  The action is irreversible.  If
taken, it succeeds with probability Q when theta=+1 and 1-Q when theta=-1
(residual risk: even a well-founded action can fail).  Success pays R_OK,
failure pays R_BAD; not acting pays 0; each step of waiting costs C (<0).
Horizon H.  The person may at any step act, give up, or wait one more step.

Person.  Each person decides by finite-horizon dynamic programming (DP) on the
payoffs *as they perceive them*:
  calibrated     : perceives correctly.
  overconfident  : believes each observation is gamma times as informative as it
                   is (perceived log-odds = gamma * true log-odds).  Belief error.
  anxious        : each step of waiting costs an extra C_anx.  Cost error.
  fearful        : a failure after acting costs an extra R_blame.  Cost error.

Device.
  none            : the person decides alone.
  forced_wait(k)  : no decision allowed before step k (the interval, Fig 2).
  verdict(p,theta): before the first decision an independent random verdict is
                   drawn: 'go' with prob p, 'wait' with prob 1-p.  'wait' makes
                   waiting bearable: the anxiety cost is scaled by (1-theta).
                   'go' makes acting blameless: the blame cost is scaled by
                   (1-theta).  Nothing is binding: the person re-optimises by DP
                   under the relieved costs (compliance is endogenous).  A verdict
                   that merely predicts what the person would do anyway ("mirror",
                   the AI-agent configuration) is by construction identical to
                   `none` and is not simulated.
  oracle(k,p,theta): the whole procedure = forced_wait(k) + verdict(p,theta) (Table 1).
  self_rule(m)    : the person acts when their *perceived* log-odds cross the
                   calibrated person's DP threshold, after talking the checked
                   value up by m (rationalisation).  Supplement S-E only.

Outcomes are objective: wrong = acted and failed (bad state or bad luck);
avoidable = acted in the bad state (the knowable part of wrong); missed = did not
act and the state was good; payoff = material payoff + C * steps.  Psychological
costs (C_anx, R_blame) are never counted in payoff.
"""
from __future__ import annotations
import functools
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------- parameters (main text)
R_OK, R_BAD, C, H = 20.0, -100.0, -0.5, 20
Q = 0.95          # P(success | theta=+1); P(success | theta=-1) = 1-Q
SIGMA = 1.25      # observation noise

_GRID = np.linspace(-30.0, 30.0, 2401)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(31)
_GH_W = _GH_W / _GH_W.sum()


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-l))


def p_success(b, q=None):
    q = Q if q is None else q
    return b * q + (1 - b) * (1 - q)


@dataclass(frozen=True)
class Person:
    gamma: float = 1.0
    c_anx: float = 0.0
    r_blame: float = 0.0
    label: str = "calibrated"


CALIBRATED = Person()
OVERCONFIDENT = Person(gamma=2.0, label="overconfident")
ANXIOUS = Person(c_anx=1.0, label="anxious")
FEARFUL = Person(r_blame=100.0, label="fearful")
PEOPLE = (CALIBRATED, OVERCONFIDENT, ANXIOUS, FEARFUL)


# ---------------------------------------------------------------- DP policy
@functools.lru_cache(maxsize=None)
def dp_thresholds(gamma, c_anx, r_blame, sigma, r_ok=None, r_bad=None, c=None, horizon=None, q=None):
    """Finite-horizon DP on the perceived model.  Returns (upper, lower, values).
    At step t the person acts if perceived log-odds >= upper[t], gives up if
    <= lower[t], otherwise waits.  Regions are checked to be one-sided."""
    r_ok = R_OK if r_ok is None else r_ok
    r_bad = R_BAD if r_bad is None else r_bad
    c = C if c is None else c
    horizon = H if horizon is None else horizon
    q = Q if q is None else q
    l = _GRID
    b = sigmoid(l)
    sig_p = sigma / np.sqrt(gamma)
    mu_inc = 2.0 / sig_p ** 2
    step_cost = c - c_anx
    ps = p_success(b, q)
    act_val = ps * r_ok + (1 - ps) * (r_bad - r_blame)
    stop_val = np.maximum(act_val, 0.0)
    upper = np.full(horizon + 2, np.inf)
    lower = np.full(horizon + 2, -np.inf)
    V_next = stop_val.copy()
    upper[horizon] = l[np.argmax(act_val >= 0)] if (act_val >= 0).any() else np.inf
    values = {horizon: V_next}
    for t in range(horizon - 1, 0, -1):
        cont = np.zeros_like(l)
        for z, w in zip(_GH_X, _GH_W):
            cont += w * (b * np.interp(l + mu_inc * (1.0 + sig_p * z), l, V_next)
                         + (1 - b) * np.interp(l + mu_inc * (-1.0 + sig_p * z), l, V_next))
        cont += step_cost
        stop = stop_val >= cont
        V = np.where(stop, stop_val, cont)
        act_region = stop & (act_val > 0)
        quit_region = stop & (act_val <= 0)
        if act_region.any():
            upper[t] = l[np.argmax(act_region)]
            assert act_region[np.searchsorted(l, upper[t]):].all(), f"non-monotone act region t={t}"
        if quit_region.any():
            lower[t] = l[len(l) - 1 - np.argmax(quit_region[::-1])]
            assert quit_region[: np.searchsorted(l, lower[t]) + 1].all(), f"non-monotone quit region t={t}"
        V_next = V
        values[t] = V
    return upper, lower, values


def thresholds_for(person: Person, sigma=SIGMA, relief_wait=0.0, relief_go=0.0, **kw):
    return dp_thresholds(person.gamma, person.c_anx * (1.0 - relief_wait),
                         person.r_blame * (1.0 - relief_go), sigma, **kw)[:2]


# ---------------------------------------------------------------- one-step lookahead (robustness table)
def lookahead_decision(l, person: Person, sigma):
    b = sigmoid(l)
    sig_p = sigma / np.sqrt(person.gamma)
    mu_inc = 2.0 / sig_p ** 2
    ps = p_success(b)
    act_val = ps * R_OK + (1 - ps) * (R_BAD - person.r_blame)
    stop_val = np.maximum(act_val, 0.0)

    def av(lp):
        psp = p_success(sigmoid(lp))
        return np.maximum(psp * R_OK + (1 - psp) * (R_BAD - person.r_blame), 0.0)

    cont = np.zeros_like(l)
    for z, w in zip(_GH_X, _GH_W):
        cont += w * (b * av(l + mu_inc * (1 + sig_p * z)) + (1 - b) * av(l + mu_inc * (-1 + sig_p * z)))
    cont += C - person.c_anx
    stop = stop_val >= cont
    return stop & (act_val > 0), stop & (act_val <= 0)


# ---------------------------------------------------------------- simulation
@dataclass
class Result:
    n: int
    payoff: float
    payoff_se: float
    wrong: float
    avoidable: float
    missed: float
    acted: float
    steps: float

    def row(self, **extra):
        d = dict(n=self.n, payoff=self.payoff, payoff_se=self.payoff_se, wrong=self.wrong,
                 avoidable=self.avoidable, missed=self.missed, acted=self.acted, steps=self.steps)
        d.update(extra)
        return d

    def __str__(self):
        return (f"payoff {self.payoff:+6.2f}±{self.payoff_se:.2f}  wrong {100*self.wrong:5.2f}% "
                f"(avoidable {100*self.avoidable:5.2f}%)  missed {100*self.missed:5.1f}%  "
                f"acted {100*self.acted:4.1f}%  steps {self.steps:4.2f}")


def _streams(seed, n):
    ss = np.random.SeedSequence(seed)
    st, ob, vd, lk = [np.random.default_rng(s) for s in ss.spawn(4)]
    theta = st.integers(0, 2, n) * 2 - 1
    X = ob.standard_normal(size=(n, H))
    success = lk.random(n) < np.where(theta > 0, Q, 1 - Q)
    return theta, X, vd, success


def _run_core(theta, X, success, sigma, person: Person, up_go, lo_go, up_wait, lo_wait,
              go_mask, k_wait=0, self_rule=None):
    n = theta.shape[0]
    L = np.zeros(n)
    done = np.zeros(n, bool)
    acted = np.zeros(n, bool)
    steps = np.zeros(n, int)
    for t in range(1, H + 1):
        live = ~done
        if not live.any():
            break
        x = theta + sigma * X[:, t - 1]
        L = np.where(live, L + 2.0 * x / sigma ** 2, L)
        steps += live
        lp = person.gamma * L
        if self_rule is not None:
            uc, lc, m = self_rule
            lchk = lp + np.sign(lp) * m
            act = lchk >= uc[t]
            quit_ = lchk <= lc[t]
        else:
            act = lp >= np.where(go_mask, up_go[t], up_wait[t])
            quit_ = lp <= np.where(go_mask, lo_go[t], lo_wait[t])
        if t < k_wait:
            act[:] = False
            quit_[:] = False
        if t == H:
            act = lp >= np.where(go_mask, up_go[H], up_wait[H])
            quit_ = ~act
        stop = live & (act | quit_)
        acted |= stop & act
        done |= stop
    wrong = acted & ~success
    avoidable = acted & (theta < 0)
    missed = (~acted) & (theta > 0)
    pay = np.where(acted, np.where(success, R_OK, R_BAD), 0.0) + C * steps
    return dict(pay=pay, wrong=wrong, avoidable=avoidable, missed=missed, acted=acted, steps=steps)


def _summarise(d, mask=None):
    if mask is None:
        mask = np.ones(d["pay"].shape[0], bool)
    n = int(mask.sum())
    pay = d["pay"][mask]
    return Result(n=n, payoff=pay.mean(), payoff_se=pay.std(ddof=1) / np.sqrt(n),
                  wrong=d["wrong"][mask].mean(), avoidable=d["avoidable"][mask].mean(),
                  missed=d["missed"][mask].mean(), acted=d["acted"][mask].mean(),
                  steps=d["steps"][mask].mean())


def simulate(person: Person, device="none", n=40_000, sigma=SIGMA, seed=0,
             k=0, p=0.5, theta=0.0, m=0.0, split=False):
    """Run n episodes.  device in {'none','forced_wait','verdict','self_rule'}.
    With split=True and device='verdict', also return Results for the episodes
    that received 'go' and 'wait'."""
    th, X, vd, suc = _streams(seed, n)
    up0, lo0 = thresholds_for(person, sigma)
    ones = np.ones(n, bool)
    if device == "none":
        d = _run_core(th, X, suc, sigma, person, up0, lo0, up0, lo0, ones)
    elif device == "forced_wait":
        d = _run_core(th, X, suc, sigma, person, up0, lo0, up0, lo0, ones, k_wait=k)
    elif device == "self_rule":
        uc, lc = thresholds_for(CALIBRATED, sigma)
        d = _run_core(th, X, suc, sigma, person, up0, lo0, up0, lo0, ones, self_rule=(uc, lc, m))
    elif device in ("verdict", "oracle"):
        # 'oracle' = the whole procedure: an interval of k steps (the rite takes time)
        # plus the random verdict with relief.  'verdict' = k=0.
        go = vd.random(n) < p
        up_go, lo_go = thresholds_for(person, sigma, relief_go=theta)
        up_wt, lo_wt = thresholds_for(person, sigma, relief_wait=theta)
        d = _run_core(th, X, suc, sigma, person, up_go, lo_go, up_wt, lo_wt, go,
                      k_wait=(k + 1 if device == "oracle" else 0))
        if split:
            return _summarise(d), _summarise(d, go), _summarise(d, ~go)
    else:
        raise ValueError(device)
    return _summarise(d)


def simulate_lookahead(person: Person, n=40_000, sigma=SIGMA, seed=0) -> Result:
    th, X, _, suc = _streams(seed, n)
    L = np.zeros(n); done = np.zeros(n, bool); acted = np.zeros(n, bool); steps = np.zeros(n, int)
    for t in range(1, H + 1):
        live = ~done
        if not live.any():
            break
        x = th + sigma * X[:, t - 1]
        L = np.where(live, L + 2.0 * x / sigma ** 2, L)
        steps += live
        lp = person.gamma * L
        act, quit_ = lookahead_decision(lp, person, sigma)
        if t == H:
            ps = p_success(sigmoid(lp))
            act = ps * R_OK + (1 - ps) * (R_BAD - person.r_blame) > 0
            quit_ = ~act
        stop = live & (act | quit_)
        acted |= stop & act
        done |= stop
    d = dict(pay=np.where(acted, np.where(suc, R_OK, R_BAD), 0.0) + C * steps,
             wrong=acted & ~suc, avoidable=acted & (th < 0), missed=(~acted) & (th > 0), acted=acted, steps=steps)
    return _summarise(d)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    print(f"R_OK={R_OK} R_BAD={R_BAD} C={C} Q={Q} sigma={SIGMA} H={H} n={n}")
    print("\n[people alone]")
    for P in PEOPLE:
        print(f"  {P.label:>13}: {simulate(P, 'none', n=n)}   | lookahead payoff {simulate_lookahead(P, n=n).payoff:+.2f}")
    print("\n[Fig 2] overconfident (and calibrated) × forced wait k")
    for k in range(0, 6):
        r = simulate(OVERCONFIDENT, "forced_wait", n=n, k=k + 1)
        rc = simulate(CALIBRATED, "forced_wait", n=n, k=k + 1)
        print(f"  k={k}: overconf avoidable {100*r.avoidable:5.2f}% wrong {100*r.wrong:5.2f}% missed {100*r.missed:4.1f}% payoff {r.payoff:+5.2f}"
              f"  | calib avoidable {100*rc.avoidable:5.2f}% missed {100*rc.missed:4.1f}% payoff {rc.payoff:+5.2f}")
    print("\n[Fig 3] verdict p=0.5")
    for P in (ANXIOUS, FEARFUL, CALIBRATED):
        print(f"  {P.label}")
        print(f"    {'none':>14}: {simulate(P, 'none', n=n)}")
        for th in (0.0, 0.9):
            allr, gor, wtr = simulate(P, "verdict", n=n, theta=th, split=True)
            print(f"    {'random θ=%.1f' % th:>14}: {allr}")
            print(f"    {'  got go':>14}: {gor}")
            print(f"    {'  got wait':>14}: {wtr}")
