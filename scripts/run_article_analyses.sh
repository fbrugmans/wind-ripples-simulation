#!/bin/bash
# Recompute the numbers, tables and figures of the article from the runs in runs/ (after scripts/run_article_series.sh).
# Usage:  bash scripts/run_article_analyses.sh [photo.HEIC]
# Text output goes to analysis/output/, figures to analysis/figures/. The scripts are run from the runs folder.
# The field photograph of 21 September 2026 (Figure 4) is not part of this repository; give its path as argument to redo that figure
# (the conversion of the HEIC file uses 'sips', macOS only).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; A="$ROOT/analysis"; O="$A/output"; F="$A/figures"
mkdir -p "$O"; cd "$ROOT/runs" || exit 1
set -x
python3 "$A/article_numbers.py" .              > "$O/article_numbers.txt"            # Tables 3 and 4, growth and size, migration, mean profile
python3 "$A/long_run.py" . SL16 .              > "$O/long_run.txt"                   # 16-hour series (Table 4, Figure 3d)
python3 "$A/spread_vs_field.py" .              > "$O/spread_vs_field.txt"            # Tables 5 and 6
python3 "$A/contributions_per_ripple.py" .     > "$O/contributions_per_ripple.txt"   # Tables 7 and 8 (per ripple)
python3 "$A/contributions_from_means.py" .     > "$O/contributions_from_means.txt"   # Table 8 (from the means)
python3 "$A/sliding_on_off.py" . . SR0 SG0     > "$O/sliding_on_off.txt"             # wind-driven sliding on and off
python3 "$A/wind_shadow.py"                    > "$O/wind_shadow.txt"                # imposed wind shadow
python3 "$A/ripple_selection.py"               > "$O/ripple_selection.txt"           # the choice of ripples
python3 "$A/sun_shadow.py"                     > "$O/sun_shadow.txt"                 # trough at the shadow line
python3 "$A/imposed_ripplets.py"               > "$O/imposed_ripplets.txt"           # imposed ripplets
# template fits (Figures 6 and 7); the fit steps take a while and use 3 processes
python3 "$A/template_fit.py" fit_free .        # free fit with the adjusted template (two trough levels, lee tail)
python3 "$A/template_fit.py" fit_fixed .
python3 "$A/template_fit.py" fit_chain .
python3 "$A/template_fit.py" fit_mirror .
python3 "$A/template_fit.py" table_mirror
# figures
python3 "$F/fig1_cross_section.py"
python3 "$F/fig3_model_ripples.py" . "$F/fig3_model_ripples.png" . SL16
python3 "$F/fig5_true_scale.py" .
python3 "$A/template_fit.py" fig6 .
python3 "$A/template_fit.py" fig7
[ -n "$1" ] && python3 "$F/fig4_photo_test.py" "$1"
# Figure 2 is a photograph (analysis/figures/fig2_grains.jpeg)
