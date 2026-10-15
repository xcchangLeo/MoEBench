# Round 3 Change Record

## Changes made

- Reframed the abstract around the execution-anchored adaptive pipeline while preserving the requested 0.21% median UnixBench error, 15.1 s selected-probe time, and 18.1 s nominal context-inclusive budget. P95 and maximum error remain in the results rather than the abstract.
- Renamed the execution-free baseline rows as Wang-style DNN and Tousi-style MLP local proxies, and stated that they do not reproduce the cited methods.
- Preserved the supplied GPU M1 / RTX 3090 Ti result values. Clarified that 27,018.0 s is the maximum of five sums of per-profile $t_i$ fields (mean 13,774.752 s), not end-to-end wall time; proxy time and BenchScout's nominal OOF budget are also identified as different cost quantities.
- Added a matched UnixBench selector diagnostic across five hosts: learned Top-3 and three fixed Top-3 strategies selected different probes but produced identical suite predictions on all 25 held-out runs. A context-only control had comparable mean error (0.61% vs. 0.59%). A focused M5 fold audit found zero probe-feature split gain in the suite output head. The paper now states that these results do not establish an accuracy advantage for learned routing.
- Kept policy OOF reports, xi-only controls, hashes, and the M5 sensitivity audit under `round-2-unixbench-policies/`.

## Verification

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded; current PDF is 12 pages. Nonfatal layout and PDF-string warnings remain.
- The original submission ZIP was not overwritten. The editable source and compiled PDF are in `work_submission/`.

## Remaining blockers

- One collection session with five runs per host-suite pair; no independent session/time/configuration holdout or locked test.
- The policy diagnostic shows no learned-selector benefit and near-equal context-only error; a probe-sensitive suite reconstructor and independent validation are needed to substantiate the routing contribution.
- CPU table historical time cells and some cross-suite costs still lack complete per-cell elapsed-time provenance and aggregation details.
- The baseline set is limited to local proxy variants; alternative Tousi-inspired models in the existing artifact achieve much stronger GPU results than the MLP shown.
- The submission bundle still lacks raw data or a stable access path, checkpoints, portable split manifests, environment details, and a complete table-generation workflow.
- PTS aggregation is a paper-defined log-mean proxy, not the official PTS score.

The independent reviewer continues to assess the submission as below WWW acceptance standard; no acceptance claim is made from manuscript edits alone.
