# Prospective Buffer-Pool SLO Feasibility Calibration

## Objective

Determine whether the existing CloudSuite Web Serving workload has any
scale-20, native-SLO-feasible PHP-FPM configuration on this host after a
bounded database buffer-pool change. This is a feasibility gate only. It does
not estimate profile-selection benefit, configuration-choice benefit, or
production generality.

## Frozen Conditions

- Workload: existing CloudSuite Web Serving / Faban usergen, scale 20.
- Cell: one 120-second full-reference steady phase, with the existing 5-second
  ramp-up and 5-second ramp-down, fixed 1000--1500 ms think time, and no probe
  cells.
- PHP-FPM: `max_children=4` and `max_children=8`, using the existing web
  bootstrap and images.
- MariaDB: the existing seeded `cloudsuite/round8-db-seeded:ready` image,
  restarted from the image for every cell. The only DB variable changed is
  `innodb_buffer_pool_size`: 134217728 bytes (128 MiB) or 2147483648 bytes
  (2 GiB). No prewarming is added.
- Other services, operation mix, dataset, host, Faban client, container
  network, and setup are held to the existing CloudSuite runner protocol.
- Native Faban per-operation limits remain p95 <= 500 ms and p99 <= 1000 ms.
  No threshold or operation-specific exception is introduced.

## Repeats and Order

Collect three independent repeats of each of the four conditions. Each cell
restarts MariaDB, Memcached, and the Web container from their frozen images.
The four conditions are run once per repeat block in the following
pre-randomized order (Python `random.Random(20261013).sample`):

| Repeat | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | 2 GiB / fpm8 | 128 MiB / fpm4 | 128 MiB / fpm8 | 2 GiB / fpm4 |
| 2 | 128 MiB / fpm4 | 128 MiB / fpm8 | 2 GiB / fpm8 | 2 GiB / fpm4 |
| 3 | 128 MiB / fpm4 | 2 GiB / fpm8 | 2 GiB / fpm4 | 128 MiB / fpm8 |

The order is frozen before calibration outcome collection. No condition is
repeated or dropped based on interim latency results.

## Cell Validity

A cell is valid only if the client exits successfully with no protocol error,
exactly one structurally valid Faban summary is retained, the prescribed
operation mix is valid, and all operations have zero transaction failures.
The server-reported buffer-pool size must equal the assigned condition.
Invalid cells remain in the raw record and count as failures for the
continuation decision; there is no replacement run.

## Continuation and Stop Rules

Continue to a larger prospective profile study only if at least one fixed
buffer-pool / PHP-FPM condition has all three independent scale-20 cells valid
and every operation passes both native p95 and p99 SLOs in each of those
three cells. Otherwise stop after these 12 cells. A failed or mixed result
does not authorize loosening the SLO, filtering operations, or adding more
loads to search for a passing point.

If the gate passes, any subsequent profile experiment must be separately
predeclared, use the passing environment as a fixed condition, and collect
new data. It may compare K=0, fixed K=1, fixed K=3, and adaptive K=3 with
matched costs. This calibration itself makes no claim about those methods.

## Provenance and Isolation

The dedicated runner and result directory use the `bsr8bpcal` name prefix.
Every raw client log, Faban XML, validation JSON, container log/inspect/diff,
condition order, image digest, runner hash, and per-cell timing is retained in
a new result directory. No original CloudSuite result directory, manifest,
paper, or submission archive is modified.
