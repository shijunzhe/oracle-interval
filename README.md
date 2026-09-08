# oracle-interval

Code, results and figures for the paper *Delaying Commitment, Deflecting Blame: The Decision Logic of Divination and the Interval AI Agents Remove*.

The paper asks what an oracle did for a person facing a decision that could not be undone, and answers with a small model: one irreversible decision, four kinds of person, and two devices, an imposed wait and a random verdict. This repository contains the model, the scripts that produce every number in the paper, the result tables, and the figures.

## Reproduce everything

```
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the tests, the main-text simulations (Figures 1 to 3 and Table 2), the supplementary simulations (Figures S1 to S4 and Tables S1 to S9), and then redraws all figures. With the default of 100,000 decisions per setting (60,000 for the supplementary sweeps) it takes under a minute on a laptop. All random draws use fixed seeds, so the tables in `results/` are regenerated exactly.

For a quick check, `python3 code/interval.py` runs the model alone with 40,000 decisions per setting in a few seconds.

## What is where

| Path | Contents |
|---|---|
| `code/interval.py` | The model: world, four people, devices, dynamic programming, simulation. The docstring at the top explains it in one page. |
| `code/test_interval.py` | Twenty-four tests of the model's stated properties (for example: the empty verdict changes nothing; full relief brings a person to the calibrated rate and no further; a verdict read as evidence erodes the relief; a mirror read as evidence raises avoidable catastrophes). |
| `code/make_figs.py` | Simulations behind Figure 1 (one trajectory), Figure 2 (imposed waiting), Figure 3 (one draw, two directions) and Table 3 (the whole procedure, the agent-timed prompt and the single-observation limit). Writes `results/`. |
| `code/make_supp.py` | Simulations behind the Supplementary Information: stakes sweep, verdict-probability sweep, θ curve, pressure strength, the self-imposed rule, subjective costs, the verdict read as evidence, the mirror verdict, two faults in one person, robustness. Writes `results/`. |
| `code/plot_figs.py`, `code/plot_supp.py` | Redraw all figures from `results/*.csv` without rerunning the simulations. Write `figs/` (300 dpi PNG and vector PDF). |
| `results/` | One CSV per figure or table. `table1.csv` is the paper's Table 3 (the paper's Table 1 is the list of persons and Table 2 the parameters and their sources); `table1_paired.csv` holds the per-decision 95% intervals of its population cells and the paired differences against own judgement. Seeds: Fig. 1 rng(3); Fig. 2 seed 11; Fig. 3 seed 23; Table 3 seed 31; S-A 41, S-B 43, S-W 45, S-D 47, S-E 49, S-F 51, S-G/S-H/S-M/S-N 23, S-P 31, robustness 53. |
| `figs/` | The figures as they appear in the paper. |

## The model in one paragraph

The world is in one of two states, good or bad, equally likely at the start. Each step brings one noisy observation, and the person keeps a running log-odds that the state is good. At any step the person may act, give up, or wait; acting is final, and even a well-founded action fails one time in twenty. Success pays 20, failure −100, giving up 0, and each step of waiting costs 0.5. Every person chooses by dynamic programming on the payoffs as they perceive them. The calibrated person perceives correctly; the overconfident person reads each observation as twice as informative as it is; the anxious person feels each step of waiting as costing three times what it does; the fearful person feels a failure after acting as costing twice what it does. An imposed wait forbids any decision before step k. A random verdict, drawn before the first decision without regard to the state or the evidence, says "go" one time in four; "wait" lifts a share θ of the anxious person's extra cost, "go" lifts the same share of the fearful person's blame, and after the verdict each person re-optimises, so compliance is a choice. Optionally the person also reads the verdict as evidence (a shift δ in perceived log-odds), or the verdict is a mirror of the person's own leaning after k observations. An agent-timed prompt lets an agent with its own γ choose the stopping step and makes the person decide on the evidence in hand. Reported outcomes are objective: avoidable catastrophes (acted in the bad state), missed opportunities (did not act in the good state), and average payoff, over 100,000 decisions per setting.

## Main results

| Result | Where |
|---|---|
| Three imposed steps cut the overconfident person's (γ = 4) avoidable catastrophes from 4.5% to 1.5% and missed opportunities from 7.3% to 2.1%, and triple their payoff (1.04 to 3.28); the calibrated person pays for every step (4.75 to 4.57 at two) | `results/fig2.csv`, Figure 2 |
| One random verdict (θ = 0.7) brings the anxious person who drew "wait" from 10.7% to 4.0% missed opportunities and the fearful person who drew "go" from 13.8% to 2.7%, against 1.8% for the calibrated person; the other verdict leaves each unchanged | `results/fig3.csv`, Figure 3 |
| The whole procedure nearly halves avoidable catastrophes (1.39% to 0.74%) and cuts missed opportunities by two thirds (8.4% to 2.8%) for an equal mix of the four persons, raising the average payoff by 0.71 ± 0.05 (3.26 to 3.97); the overconfident gain 1.89, the anxious 0.60, the fearful 0.53, the calibrated lose 0.18; an agent-timed decision (agent γ = 4) lowers the average to 0.95 | `results/table1.csv`, `results/table1_paired.csv`, Table 3 |
| Every outcome is a straight line in the probability of a "go" verdict, by construction: a fixed verdict does at least as well as a random one inside the model | `results/sB_p.csv`, Supplementary Fig. S2 |
| Discounting one's own certainty by the factor that undoes one's over-reading (3.57 for γ = 4, 5.25 for γ = 6) makes the person calibrated in every measure, but the factor is the unknown; one factor for all (1.25) leaves the γ = 4 person at 3.5% avoidable catastrophes where the wait leaves them at 2.1% | `results/sE_selfrule.csv`, Supplementary Table S1 |
| A verdict read as evidence erodes the relief: a shift of one perceived log-odds unit turns "wait" into a loss for the fearful person and leaves them worse off on average than with no oracle (3.06 against 3.71); the anxious person's relief survives one unit and is gone at two | `results/sG_evidence.csv`, Supplementary Table S4 |
| A verdict that predicts the person, read as evidence, raises avoidable catastrophes to 1.1 to 1.7 times what a random verdict read the same way and saying "go" as often leaves | `results/sH_mirror.csv`, Supplementary Table S5 |
| "Go" relief given to a person who is both overconfident (γ = 4) and afraid of blame (R_blame = 200) raises their avoidable catastrophes (3.2% to 3.7%) and lowers their payoff (1.86 to 1.52) | `results/sM_moral_hazard.csv`, Supplementary Table S6 |
| When a well-founded action cannot fail (Q = 1), fear of blame leaves missed opportunities at the calibrated rate (1.2% against 1.2%) but still delays action; a rule-of-thumb decider (one-step lookahead) gains more from the wait than the dynamic programme does | `results/robustness.csv`, Supplementary Table S2 |
| Two controls: an imposed wait in which no observation arrives changes no error rate and costs the two steps (payoff about 1.0 lower for everyone); protection of both action and inaction, given to everyone with no draw and with the same two-step interval, matches the procedure's payoff for the calibrated and overconfident and beats it for the anxious (4.43 against 4.36) and the fearful (4.51 against 4.22) | `results/sN_controls.csv`, Supplementary Table S8 |

## Requirements

Python 3.10 or later, `numpy`, `matplotlib`, `pytest`. Tested with Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9 (pinned in `requirements.txt`, because numpy's random streams are not guaranteed identical across versions).

## License

MIT. See `LICENSE`.
