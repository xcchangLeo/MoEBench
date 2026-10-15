# Session 13 Strict-Cell Record

Session 13 used probe loads 20/40/100 and full-reference loads 20/40/60/80/100, all with 60-second steady cells and full-first order. Probe throughputs were 12.25, 11.033, and 16.417 ops/s at loads 20, 40, and 100. Full throughputs were 11.6, 10.15, 9.35, 14.05, and 8.733 ops/s at loads 20, 40, 60, 80, and 100.

The first seven cells were strictly valid. Full load 100 had `mix_valid=false` and is excluded. Interpolation from probes 40 and 100 predicts 13.725 and 15.071 ops/s at loads 60 and 80, giving 27.03% MAPE over those two valid omitted-load references. A complete five-load profile metric is not reported. Valid-cell phase totals are 474.841 s for probes and 942.692 s for full cells.
