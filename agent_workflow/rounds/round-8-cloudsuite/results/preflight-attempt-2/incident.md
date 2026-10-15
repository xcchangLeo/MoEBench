# Preflight Attempt 2

No Web workload ran. The isolated database container began the official seed
download, but after about one minute wget reported approximately 61 KB/s and
an estimated 1 hour 45 minute completion time. At that point `/backup` was
about 104 MB. The preflight was stopped before SQL or schema validation because
the upstream seed transfer was too slow for the planned experiment window.

The runner's cleanup tried to export logs and metadata, but Docker reported
`No such container` before those artifacts could be read. The captured final
log is only that Docker error; no database seed completion is claimed. No
CloudSuite Faban client or Web workload was started. The first attempt, which
also failed before workload execution, is documented in the sibling directory.

The four images are present locally and have recorded digests, but the image's
database entrypoint downloads the ELGG seed on every fresh container start.
The CloudSuite multi-tier workload is therefore not used for the round-8
measurements. The experiment continues with an isolated local HTTP server and
explicitly does not claim CloudSuite coverage.
