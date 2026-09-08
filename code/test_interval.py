"""Regression tests for interval.py (T1.4).  Run: pytest -q code/test_interval.py"""
import numpy as np
import pytest
import interval as iv

N = 30_000


def se_diff(a, b, n=N):
    """crude SE of a difference of two proportions"""
    return np.sqrt(a * (1 - a) / n + b * (1 - b) / n) + 1e-9


# ---------------------------------------------------------------- DP sanity
def test_thresholds_monotone_and_stationary():
    up, lo, _ = iv.dp_thresholds(1.0, 0.0, 0.0, iv.SIGMA)
    assert np.all(np.isfinite(up[1:iv.H + 1]))
    assert up[1] > 0 and lo[1] < up[1]
    # far from the horizon the policy is stationary
    assert abs(up[1] - up[5]) < 1e-6 and abs(lo[1] - lo[5]) < 1e-6


def test_perceived_distortions_move_thresholds_the_right_way():
    up_c, lo_c, _ = iv.dp_thresholds(1.0, 0.0, 0.0, iv.SIGMA)
    up_a, lo_a, _ = iv.dp_thresholds(1.0, 1.0, 0.0, iv.SIGMA)     # anxious
    up_f, lo_f, _ = iv.dp_thresholds(1.0, 0.0, 100.0, iv.SIGMA)   # fearful
    assert up_a[1] < up_c[1] and lo_a[1] > lo_c[1]   # anxious closes earlier both ways
    assert up_f[1] > up_c[1]                         # fearful demands more certainty to act


def test_dp_beats_one_step_lookahead_for_calibrated():
    r_dp = iv.simulate(iv.CALIBRATED, "none", n=N)
    r_la = iv.simulate_lookahead(iv.CALIBRATED, n=N)
    assert r_dp.payoff > r_la.payoff + 2 * np.hypot(r_dp.payoff_se, r_la.payoff_se)


def test_calibrated_is_objectively_best_among_people():
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    for P in (iv.OVERCONFIDENT, iv.ANXIOUS, iv.FEARFUL):
        r = iv.simulate(P, "none", n=N)
        assert rc.payoff >= r.payoff - 2 * np.hypot(rc.payoff_se, r.payoff_se), P.label


# ---------------------------------------------------------------- people
def test_overconfident_makes_more_avoidable_errors():
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    ro = iv.simulate(iv.OVERCONFIDENT, "none", n=N)
    assert ro.avoidable > 3 * rc.avoidable and ro.steps < rc.steps


def test_anxious_and_fearful_miss_more():
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    for P in (iv.ANXIOUS, iv.FEARFUL):
        r = iv.simulate(P, "none", n=N)
        assert r.missed > rc.missed + 4 * se_diff(r.missed, rc.missed), P.label
        assert r.acted < rc.acted


# ---------------------------------------------------------------- forced wait (Fig 2)
def test_forced_wait_cuts_avoidable_errors_for_overconfident():
    r0 = iv.simulate(iv.OVERCONFIDENT, "forced_wait", n=N, k=1)
    r2 = iv.simulate(iv.OVERCONFIDENT, "forced_wait", n=N, k=3)
    assert r2.avoidable < 0.7 * r0.avoidable
    assert r2.missed < r0.missed


def test_forced_wait_costs_the_calibrated():
    r0 = iv.simulate(iv.CALIBRATED, "forced_wait", n=N, k=1)
    r2 = iv.simulate(iv.CALIBRATED, "forced_wait", n=N, k=3)
    assert r2.payoff < r0.payoff


def test_forced_wait_k1_equals_none():
    a = iv.simulate(iv.OVERCONFIDENT, "none", n=N)
    b = iv.simulate(iv.OVERCONFIDENT, "forced_wait", n=N, k=1)
    assert a.payoff == b.payoff and a.missed == b.missed


# ---------------------------------------------------------------- verdict (Fig 3)
def test_verdict_without_relief_changes_nothing():
    for P in (iv.ANXIOUS, iv.FEARFUL, iv.CALIBRATED):
        a = iv.simulate(P, "none", n=N)
        b = iv.simulate(P, "verdict", n=N, theta=0.0)
        assert a.payoff == b.payoff and a.missed == b.missed, P.label


def test_verdict_leaves_calibrated_unchanged():
    a = iv.simulate(iv.CALIBRATED, "none", n=N)
    b = iv.simulate(iv.CALIBRATED, "verdict", n=N, theta=0.9)
    assert a.payoff == b.payoff and a.missed == b.missed


def test_one_draw_two_directions():
    """'wait' helps the anxious and not the fearful; 'go' helps the fearful and not the anxious."""
    _, an_go, an_wait = iv.simulate(iv.ANXIOUS, "verdict", n=N, theta=0.9, split=True)
    _, an_go0, an_wait0 = iv.simulate(iv.ANXIOUS, "verdict", n=N, theta=0.0, split=True)
    _, fe_go, fe_wait = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=0.9, split=True)
    _, fe_go0, fe_wait0 = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=0.0, split=True)
    n2 = N // 2
    assert an_wait.missed < an_wait0.missed - 4 * se_diff(an_wait.missed, an_wait0.missed, n2)
    assert abs(an_go.missed - an_go0.missed) < 1e-12
    assert fe_go.missed < fe_go0.missed - 4 * se_diff(fe_go.missed, fe_go0.missed, n2)
    assert abs(fe_wait.missed - fe_wait0.missed) < 1e-12


def test_relieved_person_approaches_calibrated():
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    _, _, an_wait = iv.simulate(iv.ANXIOUS, "verdict", n=N, theta=1.0, split=True)
    _, fe_go, _ = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=1.0, split=True)
    # full relief -> same DP as calibrated -> same decisions on the same evidence
    assert abs(an_wait.missed - iv.simulate(iv.CALIBRATED, "verdict", n=N, theta=1.0, split=True)[2].missed) < 1e-12
    assert abs(fe_go.missed - iv.simulate(iv.CALIBRATED, "verdict", n=N, theta=1.0, split=True)[1].missed) < 1e-12


def test_fearful_on_go_acts_only_when_evidence_supports():
    """the 'go' verdict is permission, not command: acted rate under full relief equals the
    calibrated rate, never 100%"""
    _, fe_go, _ = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=1.0, split=True)
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    assert abs(fe_go.acted - rc.acted) < 0.02 and fe_go.acted < 0.6


# ---------------------------------------------------------------- self rule (S-E)
def test_self_rule_needs_the_factor_the_person_does_not_know():
    """discounting one's own certainty by the factor that restores the calibrated bar in
    true units brings both errors to the calibrated rates; the factor grows with gamma;
    the calibrated person who applies a discount only loses"""
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    up_cal, _ = iv.thresholds_for(iv.CALIBRATED)
    fstars = []
    for g in (1.5, 2.0, 3.0):
        P = iv.Person(gamma=g)
        up_own, _ = iv.thresholds_for(P)
        f_star = g * up_cal[1] / up_own[1]
        fstars.append(f_star)
        rs = iv.simulate(P, "self_rule", n=N, f=f_star)
        assert abs(rs.avoidable - rc.avoidable) < 3 * se_diff(rs.avoidable, rc.avoidable)
        assert abs(rs.missed - rc.missed) < 3 * se_diff(rs.missed, rc.missed) + 0.004
    assert fstars[0] < fstars[1] < fstars[2]
    r = iv.simulate(iv.CALIBRATED, "self_rule", n=N, f=1.5)
    assert r.payoff < rc.payoff


def test_thresholds_key_on_module_globals():
    """changing iv.C must not return stale cached thresholds"""
    up1, _ = iv.thresholds_for(iv.CALIBRATED)
    old = iv.C
    iv.C = -1.0
    up2, _ = iv.thresholds_for(iv.CALIBRATED)
    iv.C = old
    assert up2[1] < up1[1]


# ---------------------------------------------------------------- evidence channel and mirror (S-G, S-H)
def test_verdict_read_as_evidence_erodes_the_relief():
    own = iv.simulate(iv.FEARFUL, "none", n=N)
    r0 = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=0.9, p=0.25, delta=0.0)
    r1 = iv.simulate(iv.FEARFUL, "verdict", n=N, theta=0.9, p=0.25, delta=1.0)
    assert r0.payoff >= own.payoff - 2 * np.hypot(r0.payoff_se, own.payoff_se)
    assert r1.payoff < own.payoff - 2 * np.hypot(r1.payoff_se, own.payoff_se)


def test_mirror_equals_random_without_evidence_channel_for_unpressured_people():
    for P in (iv.CALIBRATED, iv.OVERCONFIDENT):
        rnd = iv.simulate(P, "oracle", n=N, k=2, theta=0.9, p=0.25, delta=0.0)
        mir = iv.simulate(P, "mirror", n=N, k=2, theta=0.9, delta=0.0)
        assert rnd.payoff == mir.payoff and rnd.avoidable == mir.avoidable


def test_mirror_read_as_evidence_raises_avoidable_catastrophes():
    for P in iv.PEOPLE:
        rnd = iv.simulate(P, "oracle", n=N, k=2, theta=0.9, p=0.25, delta=1.0)
        mir = iv.simulate(P, "mirror", n=N, k=2, theta=0.9, delta=1.0)
        assert mir.avoidable > rnd.avoidable + 2 * se_diff(mir.avoidable, rnd.avoidable)


# ---------------------------------------------------------------- moral hazard (S-M) and the agent-timed prompt
def test_go_relief_hurts_a_person_who_is_also_overconfident():
    P = iv.Person(gamma=2.0, r_blame=100.0)
    own = iv.simulate(P, "none", n=N)
    _, go, _ = iv.simulate(P, "verdict", n=N, theta=0.9, p=0.25, split=True)
    assert go.avoidable > own.avoidable + 2 * se_diff(go.avoidable, own.avoidable, n=N // 4)


def test_agent_timed_prompt_hands_the_calibrated_person_the_agent_error_rate():
    rc = iv.simulate(iv.CALIBRATED, "none", n=N)
    ro = iv.simulate(iv.OVERCONFIDENT, "none", n=N)
    rp = iv.simulate_prompt(iv.CALIBRATED, agent_gamma=iv.GAMMA_OVER, n=N)
    # the agent stops before the evidence justifies acting for a calibrated person, who must
    # then choose between a bad bet and giving up: errors of both kinds rise and payoff falls
    assert rp.avoidable > 3 * rc.avoidable and rp.missed > 3 * rc.missed
    assert rp.payoff < rc.payoff - 1.0
    # with a mildly overconfident agent (gamma 2) the calibrated person inherits its record
    ro2 = iv.simulate(iv.Person(gamma=2.0), "none", n=N)
    rp2 = iv.simulate_prompt(iv.CALIBRATED, agent_gamma=2.0, n=N)
    assert abs(rp2.avoidable - ro2.avoidable) < 3 * se_diff(rp2.avoidable, ro2.avoidable)
    rp1 = iv.simulate_prompt(iv.CALIBRATED, agent_gamma=1.0, n=N)
    assert abs(rp1.payoff - rc.payoff) < 3 * np.hypot(rp1.payoff_se, rc.payoff_se)


# ---------------------------------------------------------------- reproducibility
def test_seed_reproducible():
    a = iv.simulate(iv.ANXIOUS, "verdict", n=5000, theta=0.9, seed=7)
    b = iv.simulate(iv.ANXIOUS, "verdict", n=5000, theta=0.9, seed=7)
    assert a.payoff == b.payoff


def test_blind_wait_matches_none_when_no_interval():
    """blind_wait with k<=1 imposes nothing and must reproduce own judgement exactly."""
    for P in iv.PEOPLE:
        a = iv.simulate(P, "none", n=3000, seed=5)
        b = iv.simulate(P, "blind_wait", n=3000, seed=5, k=1)
        assert a.payoff == b.payoff and a.avoidable == b.avoidable and a.missed == b.missed


def test_blind_wait_charges_the_interval_and_changes_no_first_decision():
    """With k=3 and no observation in steps 2-3, the first decision is taken on the
    step-1 observation alone, so every episode runs at least three steps."""
    own = iv.simulate(iv.OVERCONFIDENT, "none", n=3000, seed=5)
    r = iv.simulate(iv.OVERCONFIDENT, "blind_wait", n=3000, seed=5, k=3)
    assert r.steps >= 3.0
    # same observations in the same order, so the same decisions, two steps later
    assert abs(r.avoidable - own.avoidable) < 1e-9 and abs(r.missed - own.missed) < 2e-3
    assert abs((own.payoff - r.payoff) - 2 * (-iv.C)) < 0.05
