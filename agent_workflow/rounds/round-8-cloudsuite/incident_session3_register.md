# Session 3 Full-Reference Operation Failure

During the original 80-cell collection, the `fpm8/session3/full/scale40`
Faban summary recorded one failed `Register` operation among 3,999 attempted
operations (3,998 successes, failure rate 0.025%). The client log records an
HTTP 400 response from `/uservalidationbyemail/emailsent`. The XML and
validation JSON are retained in the original result directory.

The analysis contract set before collection requires zero transaction failures
for a throughput cell to be usable. This cell is therefore invalid for the
strict five-load profile endpoint, irrespective of its measured throughput
(33.317 operations/s). The original run remains intact. A separately labeled
fifth paired session, if collected, is a protocol-validity replacement; the
analysis must use a declared validity rule rather than selecting sessions by
their error or configuration outcome. Any relaxed analysis that includes the
failed cell is sensitivity analysis and must disclose the failure.
