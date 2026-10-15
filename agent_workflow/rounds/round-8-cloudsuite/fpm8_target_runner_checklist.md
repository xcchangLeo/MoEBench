# fpm8 Target-Session Runner Checklist

Prepare this runner only after the current CloudSuite run is complete and its
containers have been cleaned up. The runner is for four new fpm8 sessions; it
must not connect to, inspect, stop, restart, or remove the current `bsr8`
collector or any `live-kbench` container.

## Freeze Gate

- Require an explicit, read-only frozen manifest path. Refuse to start if it
  is absent, invalid, or does not have a recorded SHA-256 digest.
- The manifest must identify the four complete fpm4 source sessions and hash
  every source full-reference summary and validation record used for training
  or history prediction. Reject any manifest that references an fpm8 record
  from the current CloudSuite run.
- Record the chosen three load scales, 60-second probe duration, 120-second
  reference duration, predictor method, model/preprocessing versions,
  hyperparameters, predictor source hash, training-data hashes, and locked
  load-order seed. Validate exactly three distinct scales from
  `{20, 40, 60, 80, 100}`.
- Validate all four fpm4 source sessions and their five full-reference cells
  before creating Docker resources. A failed freeze gate must have no Docker
  side effects.

## Isolation and Naming

- Create a unique run prefix such as `bsr8-change-<UTC timestamp>-<pid>` and
  put all outputs below a separate `results/change-<prefix>/` directory.
- Use a private network with that prefix and prefix every web, database,
  Memcached, and Faban client container name with it. Save the created network
  ID and each container ID.
- Keep an in-memory list of only containers created by this process. Cleanup
  may archive/remove only those exact names and remove the network only when
  its saved ID matches the network created by this process. Do not use global
  `docker rm`, `docker stop`, `docker prune`, name-prefix sweeps, or cleanup
  against shared containers.
- Before creation, fail on any exact name collision. Do not inspect unrelated
  container names to find a free name.

## Per-Session State Machine

Use globally unique target-session IDs 101--104, distinct from current and
supplemental session IDs.
Keep the order `probe -> prediction lock -> full reference` fixed for every
session.

1. Record session metadata, host load, image digests, the frozen manifest
   digest, and the randomized three-load probe order.
2. Restore and validate the seeded database, start Memcached and the fpm8 web
   tier, and run only the three manifest-selected 60-second probes. Randomize
   order using the locked seed. Copy each complete Faban output tree, raw
   `summary.xml`, client stdout/log, container inspect/log/diff, validation
   JSON, timings, and IDs before cleaning up that cell's containers.
3. Validate all three probe cells. Preserve invalid cells and reasons. Do not
   continue to the full phase if a probe has a client protocol error, invalid
   XML, or invalid operation mix; mark the session incomplete and retain its
   evidence.
4. Invoke only the predictor whose source hash and model ID match the frozen
   manifest. Write the complete five-load throughput prediction and metadata
   atomically. Hash the prediction file and a provenance record containing
   the selected probe cell hashes, model/manifest hashes, command, and
   completion timestamp. Verify the hash from disk before allowing the full
   phase to begin. Predictor failure or a missing hash must abort the session
   before any full-reference cell starts.
5. Only after the prediction lock, run the five 120-second fpm8 full-reference
   cells in an independently randomized load order. Archive and validate raw
   outputs exactly as for probes. Never rewrite the locked prediction.
6. Record phase and per-cell start/end times, setup duration, runner duration,
   planned steady time, exact realized order, and failure status. Do not update
   model parameters or selected loads between sessions.

## Result and Failure Records

- Keep `runs.jsonl`, `phase-times.jsonl`, `cell-times.jsonl`, and
  `randomized-order.txt` in the separate result directory. Store raw Faban XML
  and all validator outputs by session, phase, and load; hash each copied XML.
- Treat mix-valid, zero-operation-failure throughput as usable even when
  Faban reports an SLO/latency failure. Preserve `benchSummary` status, each
  operation's p99/sample count/censor state, operation-mix status, and failed
  operations as distinct outcomes. Never abort or relabel a valid throughput
  cell solely because its latency SLO fails.
- Score the frozen primary endpoint only after all prediction files and hashes
  are locked. Report per-session five-load throughput MAPE and paired
  history-only comparison, with sessions as the independent unit. Keep
  `BrowsetoElgg` p99 descriptive and report each cell's successful sample
  count; do not form an overall request-mixture p99 or a QoS claim from
  throughput ranking.
- On interruption, archive and clean only this run's recorded containers and
  leave a non-complete status marker. Resume only from a new, separately
  identified run after checking the frozen manifest and prior evidence.

## Go/No-Go Before Implementation

- Wait for the current CloudSuite collector to finish; do not run this runner
  concurrently with it.
- Finalize and review the frozen manifest schema and prediction command/API.
  The runner must not infer a model or subset from target fpm8 observations.
- Confirm the existing fpm4 source set has four independent sessions with
  complete and valid full-reference curves. Keep the expected fpm8 SLO failure
  rate visible in the analysis; the current first-session evidence has valid
  operation mix and zero transaction failures, but no cells passed the
  latency SLO.
- Perform syntax, manifest-validation, and dry-run resource-name checks before
  any workload invocation. Do not use a workload smoke test until a separate
  post-CloudSuite window is explicitly scheduled.
