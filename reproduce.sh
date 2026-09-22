#!/usr/bin/env bash
# Reproduces every number and figure in the paper. About half a minute at the default
# of 100,000 decisions per setting; pass a smaller number for a quick run:  bash reproduce.sh 20000 20000
set -e
cd "$(dirname "$0")/code"
N="${1:-100000}"
python3 -m pytest -q test_interval.py
python3 make_figs.py "$N"          # Figures 1-3, Tables 3 and 4, Tables S7, S10, S11  ->  ../results, ../figs
python3 make_supp.py "${2:-60000}"  # Supplementary figures and tables (60,000 per setting by default)  ->  ../results, ../figs
python3 plot_figs.py               # redraw Figures 1-3 under the figure rules (300 dpi PNG + PDF)
python3 plot_supp.py               # redraw Supplementary Figs S1-S5
echo "done: results/ and figs/ regenerated with N=$N"
