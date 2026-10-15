# UnixBench Whole-Host Holdout Diagnostic

## Reproduction

```sh
PYTHONPATH=. python scripts/analyze_unixbench_host_holdout.py \
  --output agent_workflow/rounds/round-8-host-holdout/unixbench_oracle.json
```

The input is the canonical May 2026 UnixBench session for each of five hosts,
five complete runs per host. The later October M1 diagnostic and all PTS
sessions are excluded. The JSON lists the absolute path and SHA-256 of each
input run and every outer fold's training and test paths.

## Split and Methods

The outer split holds out one *entire host* (five runs) and trains on the other
four hosts (20 runs). The nested selector compares all 220 three-component
subsets using leave-one-training-host-out mean relative suite error; no held-out
host value enters subset selection. Each chosen subset feeds an alpha-1 Ridge
regression from standardized log component indices to log official UnixBench
suite index. The score-ranked subset is selected by the highest mean log
component score in the outer training hosts. A fixed first-three subset, 100
seeded random subsets, an outer-training mean without measurements, and a
context-only Ridge regressor are controls. The context regressor uses the Xi
feature vector and selects alpha from {1, 10, 100, 1000} by inner whole-host
cross-validation; standardization is fitted only on the training hosts. The
suite target is the official score recorded in the raw run.

## Result

Mean of five held-out-host mean absolute relative errors:

| Policy | Error (%) |
|---|---:|
| Score-ranked three components | 4.489 |
| Nested target-loss-selected three components | 8.083 |
| Mean of 100 random three-component subsets | 9.530 |
| Fixed first three components | 12.574 |
| Context-only Ridge, nested alpha | 16.255 |
| Training-mean, no component measurement | 22.596 |

The score-ranked policy ranges from 0.981% to 10.245% across held-out hosts.
The nested selector ranges from 0.635% to 14.425%; its training-host selection
does not transfer consistently with only four source hosts. These results
support a limited information-value observation: genuine target-host component
measurements help compared with a training mean, but which subset to measure
is unstable across unseen hosts.

This is an *oracle component-score* analysis. It supplies scores extracted
from complete target-host runs and therefore cannot validate short-probe
prediction, the end-to-end BenchScout pipeline, online wall time, or the
15-second claim. There is one collection session per host and only five host
identities; the 25 run-level errors are clustered by host, not 25 independent
transfer tests. The selector optimizes suite-index loss rather than profile
error, measurement cost, or a Web-serving decision. No policy should be
called a statistically established winner from five held-out hosts.

| Artifact | SHA-256 |
|---|---|
| `scripts/analyze_unixbench_host_holdout.py` | `3b65bc46b40dcb2438bf96f05645b2ecfd407e9d33bb4df624df2449edd44658` |
| `unixbench_oracle.json` | `10aef4a1781dd007f40eea8746ed82f854fa6f5d9a487840bc062dbf8e005b1f` |
