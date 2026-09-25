# oracle-interval

Code, results and figures for the article *Waiting, judgement, and responsibility: lessons from divination for human oversight of irreversible decisions.

The article asks what an oracle did for a person facing a decision that could not be undone, and answers with a small model: one irreversible decision, four kinds of person, an imposed wait, a random verdict, and an AI agent that hands the decision over. This repository contains the model, the scripts that produce every number in the article and its Supplementary Information, the result tables, and the figures. There are no human-participant, survey or personal data: every data file here is a simulation output that the code regenerates from fixed random seeds.

## Reproduce everything

```
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the tests, the main-text simulations (Figures 1 to 3, Table 3 and Table 4), the supplementary simulations (Supplementary Figures S1 to S5 and Tables S1 to S11), and then redraws all figures. With the default of 100,000 decisions per setting (60,000 for the supplementary sweeps) it takes under a minute on a laptop. All random draws use fixed seeds, so the tables in `results/` are regenerated exactly. `python3 code/experiments/convergence.py` reruns the main cells under finer grids and different quadrature (Supplementary Table S12).

For a quick check, `python3 code/interval.py` runs the model alone with 40,000 decisions per setting in a few seconds.

## Which file is behind which figure or table

| In the article | Result file | Produced by | Seed |
|---|---|---|---|
| Fig. 1 (one trajectory) | `results/fig1_trajectory.csv` | `make_figs.py` | rng(3) |
| Fig. 2 (imposed waiting) | `results/fig2.csv` | `make_figs.py` | 11 |
| Fig. 3 (one verdict, two pressures) | `results/fig3.csv` | `make_figs.py` | 23 |
| Table 3 (the whole procedure, agent-timed and single-observation limits) | `results/table1.csv`, `results/table1_paired.csv` (per-decision 95% intervals and paired differences); `table1_p05.csv`, `table1_paired_p05.csv` for the higher observation cost | `make_figs.py` | 31 |
| Table 4 (hand-off arrangements A–D by the agent's γ) | `results/table4_agent_gamma.csv` | `make_figs.py` | 31 |
| Supplementary Fig. S1 (stakes and observation cost) | `results/sA_stakes.csv` | `make_supp.py` | 41 |
| Supplementary Fig. S2 (probability of "go") | `results/sB_p.csv` | `make_supp.py` | 43 |
| Supplementary Fig. S3 (share of pressure relieved, θ) | `results/sD_theta.csv` | `make_supp.py` | 47 |
| Supplementary Fig. S4 (pressure strength) | `results/sF_pressure.csv` | `make_supp.py` | 51 |
| Supplementary Fig. S5 and Table S10 (hand-off by person) | `results/sS10_handoff_persons.csv` | `make_figs.py` | 31 |
| Supplementary Table S1 (self-imposed rule) | `results/sE_selfrule.csv` | `make_supp.py` | 49 |
| Supplementary Table S2 (robustness, Q = 1 and one-step lookahead) | `results/robustness.csv` | `make_supp.py` | 53 |
| Supplementary Table S4 (verdict read as evidence) | `results/sG_evidence.csv` | `make_supp.py` | 23 |
| Supplementary Table S5 (mirror verdict) | `results/sH_mirror.csv` | `make_supp.py` | 23 |
| Supplementary Table S6 (overconfident and fearful) | `results/sM_moral_hazard.csv` | `make_supp.py` | 23 |
| Supplementary Table S7 (Table 3 by person, paired) | `results/table1_paired.csv` | `make_figs.py` | 31 |
| Supplementary Table S8 (two controls) | `results/sN_controls.csv` | `make_supp.py` | 23 |
| Supplementary Table S9 (agent-timed prompt by agent γ) | `results/sP_agent.csv` | `make_supp.py` | 31 |
| Supplementary Table S11 (followers) | `results/sS11_followers.csv` | `make_figs.py` | 31 |
| Supplementary Table S12 (precision checks) | `results/convergence.csv` | `experiments/convergence.py` | as above |
| Subjective-cost columns (Supplementary Note 1) | `results/sW_subjective.csv` | `make_supp.py` | 45 |

Tables 1 and 2 of the article list the decision types and the parameter values; they have no data file. Supplementary Table S3 (parameter rationale and ranges) is documentary.

Column names in the CSV files are the quantities named in the article: `avoidable` is the share of decisions in which the person acted in the bad state, `missed` the share in which the person gave up in the good state, `wrong` the share of decisions in which an action was taken and failed, `payoff` the mean material payoff, `steps` the mean number of observations, `payoff_se` a standard error and `hw_vs_A` the paired 95% Monte Carlo half-width of the difference against arrangement A. Every rate uses all simulated decisions as its denominator unless the column name says otherwise (prefixes such as `go_` and `wait_` restrict to decisions receiving that verdict).

## What is where

| Path | Contents |
|---|---|
| `code/interval.py` | The model: world, four people, devices, dynamic programming, simulation, hand-off. The docstring at the top explains it in one page. |
| `code/test_interval.py` | Thirty tests of the model's stated properties (for example: the empty verdict changes nothing; full relief brings a person to the calibrated rate and no further; a verdict read as evidence erodes the relief; a mirror read as evidence raises actions in the bad state). |
| `code/make_figs.py` | Simulations behind Figures 1–3, Table 3, Table 4 and Supplementary Tables S7, S10, S11. Writes `results/`. |
| `code/make_supp.py` | Simulations behind the Supplementary Information: stakes sweep, verdict-probability sweep, θ curve, pressure strength, the self-imposed rule, subjective costs, the verdict read as evidence, the mirror verdict, two faults in one person, the agent-timed prompt, two controls, robustness. Writes `results/`. |
| `code/experiments/convergence.py` | Precision checks: finer evidence grids and different quadrature (Supplementary Table S12). |
| `code/plot_figs.py`, `code/plot_supp.py` | Redraw all figures from `results/*.csv` without rerunning the simulations. Write `figs/` (300 dpi PNG and vector PDF). |
| `results/` | One CSV per figure or table, as mapped above. |
| `figs/` | The figures as they appear in the article and the Supplementary Information. |

## The model in one paragraph

The world is in one of two states, good or bad, equally likely at the start. Each step brings one noisy observation, and the person keeps a running log-odds that the state is good. At any step the person may act, give up, or wait; acting is final, and even a well-founded action fails one time in twenty. Success pays 20, failure −100, giving up 0, and each observation step costs 0.5. Every person chooses by dynamic programming on the payoffs as they perceive them. The calibrated person perceives correctly; the overconfident person weights each observation four times as strongly as it should be weighted (γ = 4; 2 and 6 also shown); the anxious person feels each step of waiting as costing four times its material cost (C_anx = 1.5); the fearful person feels a failure after acting as costing three times its material loss (R_blame = 200). An imposed wait forbids any decision before k additional observations have arrived. A random verdict, drawn without regard to the state or the evidence, says "go" one time in four; "wait" lifts a share θ = 0.7 of the anxious person's extra cost, "go" lifts the same share of the fearful person's blame, and after the verdict each person re-optimises, so compliance is a choice. Optionally the person also reads the verdict as evidence (a shift δ in perceived log-odds), or the verdict is a mirror of the person's own leaning after two observations. An agent with its own weighting γ_a collects observations until its own rule would act or give up and then hands over; the person may then decide at once, continue checking, receive relief, or be required to obtain two new observations before acting, with giving up open at any time; a share f of followers simply take the agent's decision. Reported outcomes are material: actions in the bad state, missed opportunities, and average payoff, over 100,000 decisions per setting.

## Requirements

Python 3.10 or later, `numpy`, `matplotlib`, `pytest`. Tested with Python 3.11.15, numpy 2.4.4, matplotlib 3.10.9 (pinned in `requirements.txt`, because numpy's random streams are not guaranteed identical across versions).

## How to cite

Please cite the article, and the archived code release as software (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). Result files and figures: CC BY 4.0.
