# oracle-interval

Code, results and figures for the paper *Delaying Commitment, Deflecting Blame: The Decision Logic of Divination and the Interval AI Agents Remove*.

The paper asks what an oracle did for a person facing a decision that could not be undone, and answers with a small model: one irreversible decision, four kinds of person, and two devices, an imposed wait and a random verdict. This repository contains the model, the scripts that produce every number in the paper, the result tables, and the figures.

## Reproduce everything

```
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the tests, the main-text simulations (Figures 1 to 3 and Table 1), the supplementary simulations (Figures S1 to S4 and Tables S1 to S3), and then redraws all figures. With the default of 100,000 decisions per setting (60,000 for the supplementary sweeps) it takes under a minute on a laptop. All random draws use fixed seeds, so the tables in `results/` are regenerated exactly.

For a quick check, `python3 code/interval.py` runs the model alone with 40,000 decisions per setting in a few seconds.

## What is where

| Path | Contents |
|---|---|
| `code/interval.py` | The model: world, four people, devices, dynamic programming, simulation. The docstring at the top explains it in one page. |
| `code/test_interval.py` | Sixteen tests of the model's stated properties (for example: the empty verdict changes nothing; full relief brings a person to the calibrated rate and no further). |
| `code/make_figs.py` | Simulations behind Figure 1 (one trajectory), Figure 2 (imposed waiting), Figure 3 (one draw, two directions) and Table 1 (the whole procedure). Writes `results/`. |
| `code/make_supp.py` | Simulations behind the Supplementary Information: stakes sweep, verdict-probability sweep, θ curve, pressure strength, the self-imposed rule, subjective costs, robustness. Writes `results/`. |
| `code/plot_figs.py`, `code/plot_supp.py` | Redraw all figures from `results/*.csv` without rerunning the simulations. Write `figs/` (300 dpi PNG and vector PDF). |
| `results/` | One CSV per figure or table. |
| `figs/` | The figures as they appear in the paper. |

## The model in one paragraph

The world is in one of two states, good or bad, equally likely at the start. Each step brings one noisy observation, and the person keeps a running log-odds that the state is good. At any step the person may act, give up, or wait; acting is final, and even a well-founded action fails one time in twenty. Success pays 20, failure −100, giving up 0, and each step of waiting costs 0.5. Every person chooses by dynamic programming on the payoffs as they perceive them. The calibrated person perceives correctly; the overconfident person reads each observation as twice as informative as it is; the anxious person feels each step of waiting as costing three times what it does; the fearful person feels a failure after acting as costing twice what it does. An imposed wait forbids any decision before step k. A random verdict, drawn before the first decision without regard to the state or the evidence, says "go" one time in four; "wait" lifts a share θ of the anxious person's extra cost, "go" lifts the same share of the fearful person's blame, and after the verdict each person re-optimises, so compliance is a choice. Reported outcomes are objective: avoidable catastrophes (acted in the bad state), missed opportunities (did not act in the good state), and average payoff, over 100,000 decisions per setting.

## Main results

| Result | Where |
|---|---|
| Two imposed steps cut the overconfident person's avoidable catastrophes from 1.6% to 1.0%, three steps to 0.7%; missed opportunities fall at the same time; the calibrated person pays for every step | `results/fig2.csv`, Figure 2 |
| One random verdict brings the anxious person who drew "wait" from 7.4% to 2.2% missed opportunities and the fearful person who drew "go" from 3.4% to 2.0%, against 1.8% for the calibrated person; a verdict with no relief changes nothing | `results/fig3.csv`, Figure 3 |
| The whole procedure cuts avoidable catastrophes by a third and missed opportunities by more than half for a mixed population, and pays only when a step of waiting costs about 0.5% of the catastrophe or less; deciding after a single observation collapses the payoff from 4.29 to 0.38 | `results/table1.csv`, `results/sA_stakes.csv`, Table 1, Supplementary Fig. S1 |
| Every outcome is a straight line in the probability of a "go" verdict: the draw itself buys no expected payoff | `results/sB_p.csv`, Supplementary Fig. S2 |
| A self-imposed higher bar makes the overconfident person act earlier, not later | `results/sE_selfrule.csv`, Supplementary Table S2 |
| Fear of blame does no damage when a well-founded action cannot fail (Q = 1) | `results/robustness.csv`, Supplementary Table S3 |

## Requirements

Python 3.10 or later, `numpy`, `matplotlib`, `pytest`. Tested with numpy 2.4, matplotlib 3.10.

## License

MIT. See `LICENSE`.
