# Round 8 Reproduction Artifacts

This directory contains a host-held-out UnixBench oracle diagnostic. It is
included to make the negative transfer result auditable; it is not a claim
that the complete short-probe pipeline generalizes across hosts.

Run from the repository root:

```sh
PYTHONPATH=. python scripts/analyze_unixbench_host_holdout.py \
  --dataset-root dataset \
  --output agent_workflow/rounds/round-8-host-holdout/unixbench_oracle.json
```

The input records are the five May 2026 UnixBench runs for each of five host
identities. The JSON records input paths and SHA-256 hashes, outer host splits,
nested selector choices, and per-run errors. Full-run component values are
provided at the held-out host, so the result is an oracle information-value
diagnostic. It does not measure probe time or validate the end-to-end router,
probe estimator, or reconstructor.
