# Wind Ripples Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24323843.svg)](https://doi.org/10.5281/zenodo.24323843)

A two-dimensional numerical model of aeolian sand ripples: grains saltate in a wind profile that they slow down
themselves, impact, eject other grains, push bed grains forward (creep) and slide over the bed driven by the wind,
while the bed shear stress and the direction of the wind over the ripples follow from linear flow theory for a rough,
undulating bed. Ripples form from a flat bed with noise; their shape is not imposed.

This repository contains the model code and everything needed to reproduce the results of the article

> Brugmans, F. (in preparation). How do wind ripples become asymmetric? A model study of initiation, size and shape.

The model results are compared with the ripples measured on the Hulshorster Zand, the Netherlands, by
Brugmans (1983), *Earth Surface Processes and Landforms* 8, 527–534, https://doi.org/10.1002/esp.3290080604.

## Requirements

- macOS or Linux with `clang` (the C part is built automatically on first use with `clang -O3 -shared -fPIC`)
- Python 3 with NumPy, SciPy, Matplotlib and Pillow

The article runs were made on an Apple M1 with Apple clang 21, Python 3.14, NumPy 2.4, SciPy 1.17 and Matplotlib 3.10.
Each run uses one processor core.

## Quick start

```bash
python3 tests/test_reproduce.py          # about half a minute: checks that the first 60 s are bit-identical to the article run
python3 run_model.py 271                 # the article model, seed 271, 4 hours of model time (about 2 hours of computing)
python3 run_model.py 271 SR0 TEIND=600   # the same, 10 minutes of model time
python3 run_model.py 271 SG0             # the article model without wind-driven sliding
```

Output goes to `runs/`: `c23_z<seed>_<series>.npz` and a manifest `….manifest.json` with the settings, the hashes of the
source files, the software versions and the hash of the output.

Bit-identity with the article runs is expected with clang on Apple silicon. With another compiler or processor small
floating-point differences can grow over time; the runs are then statistically equivalent but not identical.

## Reproducing the article

```bash
bash scripts/run_article_series.sh       # all 37 series x 4 seeds (several days on 3 cores; restartable)
bash scripts/run_article_analyses.sh     # numbers, tables and figures from the runs
```

`JOBS=6 bash scripts/run_article_series.sh SR0 SG0` runs only some series, with 6 runs at a time. Text output of the
analyses goes to `analysis/output/`, figures to `analysis/figures/`. Which script gives which table or figure is noted in
`scripts/run_article_analyses.sh`.

| Series | Description | Used for |
| --- | --- | --- |
| SR0 | the article model: wind from the theory for a rough bed (z0 = 0.1 mm), streamlines follow the bed | all results |
| SL16 | SR0 continued to 16 hours | Table 4, Figure 3 |
| SG0 | SR0 without wind-driven sliding | Table 8, sliding on and off |
| SR5, SQ0, SQ5, SQ8, SQ3 | with an imposed wind shadow and/or without the streamlines | Tables 7 and 8, wind shadow |
| PKO, PKS, CO10, CO25, CO50 | wind with a viscous sublayer and curvature correction; cohesion | Tables 7 and 8 |
| NMO, NMS, KAO, KAS, ABO, ABS, AZO, AZS, VLO, VLS, BRLO, BRLS | earlier model versions compared in the article | Tables 7 and 8 |
| RN, RA, RB (31/50, K/L) | initiation from a flat bed (RN) and with an imposed ripplet of 35 mm (RA) or 102 mm (RB), at u* 0.31 and 0.50 m/s | Table 3, Discussion |

The complete settings of each series are in `series.py`.

## Repository structure

| Path | Contents |
| --- | --- |
| `model/wind_korrel.c`, `model/wind_korrel.py` | the grain model (C) and its Python wrapper: grains in the air, impact, creep, wind-driven sliding, avalanching, wind over the ripple |
| `model/run_fetch.py`, `model/run_factor.py` | a model run: settings, the bed of 2140 mm, records every 30 s |
| `model/ripple_measure.py` | crests, troughs and ripple geometry of a profile |
| `model/tabel_vandriest_*.npz` | tables A(k), B(k) for the wind with a viscous sublayer (only for the comparison series) |
| `run_model.py`, `series.py` | start a run; the settings of all series |
| `tests/` | the reproduction test and its reference |
| `analysis/` | the analyses of the article; `measure.py` holds the measuring rule |
| `analysis/figures/` | the figure scripts and the photograph of Figure 2 |
| `scripts/` | run all series; run all analyses |

## Settings

Settings are passed to the model code as environment variables. Their names are Dutch, as in the manifests of the
article runs; extra `NAME=value` arguments of `run_model.py` override the series settings. The most important:

| Setting | Article value | Meaning |
| --- | --- | --- |
| `USTAR` | 0.35 | shear velocity above the transport layer (m/s) |
| `TEIND` | 14400 | model time (s) |
| `OPNAME_DT` | 30 | seconds between records |
| `NAAMSUF` | `_<series>` | suffix of the output name |
| `CVAR` | C23 | grain and collision rules (three grain classes, Beladjine et al. 2007) |
| `KORRELMASSA` | 1 | a filled cell is one grain of its class (*korrel* = grain) |
| `VLAGEN` | 0 | gusts (*vlagen*) off |
| `PCR` | 0.2116 | creep: mean number of bed grains pushed per impact |
| `KRUIP_INSLAG` | 0 | creep distance scaled with the local impact rate (*kruip* = creep, *inslag* = impact) |
| `KRUIP_TERUG`, `KAATS_TERUG` | 1 | creep and rebound in the direction of motion of the impacting grain (*kaatsen* = rebound) |
| `WINDROL`, `WR_ZETA` | 2, 1e9 | wind-driven sliding on, and its rate (grains per m² per s) |
| `WAND` | 3 | a grain hitting the side of a higher column rebounds from it (*wand* = wall) |
| `CEL_TOREN` | tan | toppling rule for columns (*toren* = tower) |
| `TILT_PLUS`, `HOEK_LSQ_KOL` | 1, 12 | impact and take-off angles relative to the bed slope over 12 columns (9.6 mm) |
| `MOD_AB`, `MOD_Z0` | 1, 0.1 | wind acceleration from the theory for a fully rough bed, roughness length (mm) |
| `STROOM` | 1 | streamlines follow the bed (*stroomlijn* = streamline) |
| `OMH_GR` | – | imposed wind shadow: line at this angle (degrees) behind each crest (*omhullende* = envelope) |
| `COHESIE` | 0 | cohesion number (*cohesie*) |

The code contains further switches from the development of the model (variants and diagnostic counters). They are off
by default; with a switch off, the result is bit-identical to the run without it. Variable and function names inside the
code are partly Dutch; comments and output are in English.

## Output

Each run writes a compressed NumPy file. The main arrays (one row per record):

| Key | Contents |
| --- | --- |
| `t` | time of each record (s) |
| `h` | bed height per column, in layers; multiply by `dz` (mm) |
| `h0` | initial bed |
| `dx`, `dz` | cell size (mm) |
| `ins`, `kr` | impacts and creep departures per column in the window before the record |
| `top` | grain class of the top grain per column (0 = 176 µm, 1 = 334 µm, 2 = 488 µm) |
| `skr` | sum of the creep distances in the window (m) |
| `wr` | wind-driven sliding in the window: number of sliding grains, sum of the sliding distance (m) |
| `U` | mean wind speed of the gust series in the window (m/s; only meaningful with gusts) |

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The figures and the photograph are released under CC BY 4.0.
How to cite this software is given in `CITATION.cff`; the archived version is at Zenodo, https://doi.org/10.5281/zenodo.24323843.

The model code, the runs, the analyses and the figures were made with Claude (Anthropic), at the direction of and in
consultation with the author, who set the questions, the experiments and the choices and judged every result.
