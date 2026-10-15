# Session 9 Selective Pair Analysis

Session 9 is a matched-duration stress case for the frozen fixed-even linear-interpolation policy. It uses probe loads 20/60/100 and full-reference loads 20/40/60/80/100, with 60-second steady cells for both phases. The full-reference phase ran first, followed by the probe phase.

All eight cells are valid with zero operation failures. The omitted-load MAPE is 19.9261% and the full five-load MAPE is 14.6088%. Cell-total phase sums are 472.638 s for the three probes and 792.409 s for the five full references, a 40.35% reduction. The matched durations remove the 60-versus-120-second duration difference, but full-first ordering and possible temporal drift remain confounds; this is an exploratory stress case, not a general speedup estimate.
