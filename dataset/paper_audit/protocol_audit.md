# Paper Protocol Audit

- Generated: `2026-09-11T16:17:40.940875+08:00`
- Raw run records: `55`
- Host-suite pairs: `11`

## Dataset coverage

| Host | Suite | Sessions | Runs | Strict session holdout |
|---|---|---:|---:|---|
| aces-System-Product-Name | phoronix_cpu | 1 | 5 | False |
| aces-System-Product-Name | phoronix_gpu | 1 | 5 | False |
| aces-System-Product-Name | unixbench | 1 | 5 | False |
| iZbp15n87643uk1sqjrdvdZ | phoronix_cpu | 1 | 5 | False |
| iZbp15n87643uk1sqjrdvdZ | unixbench | 1 | 5 | False |
| iZbp16krl0yc7euw7sb6slZ | phoronix_cpu | 1 | 5 | False |
| iZbp16krl0yc7euw7sb6slZ | unixbench | 1 | 5 | False |
| iZbp1acaw5wdllhz47922rZ | phoronix_cpu | 1 | 5 | False |
| iZbp1acaw5wdllhz47922rZ | unixbench | 1 | 5 | False |
| iZbp1glgt48i9a8d49embxZ | phoronix_cpu | 1 | 5 | False |
| iZbp1glgt48i9a8d49embxZ | unixbench | 1 | 5 | False |

## Findings

- Raw datasets contain 5 observed host identities.
- Each observed host-suite pair has one raw collection session with five runs.
- The repository does not contain a machine-readable train/test manifest for the headline results.
- Session-level holdout across independently collected sessions cannot be demonstrated from the current raw data alone.

## Interpretation

The current raw corpus supports leave-one-run-out evaluation within a single collection session and can support leave-one-session-out only when multiple independent sessions exist. The audit therefore marks cross-session generalization as unavailable for the current host-suite pairs rather than inferring it from same-session runs.

Experiment reports are recorded separately because a report's CV mode does not prove that every model-selection artifact was trained without test information. The revised paper must state the split and tuning protocol explicitly.
