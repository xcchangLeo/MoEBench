# Preflight Attempt 1

No Web workload ran in this attempt. The CloudSuite database image downloaded
and extracted the EPFL `ELGG_DB.tar.gz` seed. During the download, the extracted
`/backup` directory reached approximately 2.4 GB and the wget progress reached
90%. The container was removed before seed initialization completed; no
successful authenticated SQL query or seed-image snapshot was recorded.

The runner file was edited while its process was still executing. The shell
reported `unexpected EOF while looking for matching '"'` and exited; its trap
removed the round-8 container and bridge. The observed logs were not exported
before removal. The container ID, network ID, and failed SQL watcher output are
preserved beside this note. This is a runner-process failure, not a workload
result.
