# BenchScout review-edit loop

This directory defines the repeatable review loop used for the WWW submission.
The original archive is immutable. The current manuscript lives in
`work_submission/`, and each review/experiment round records its evidence in
`agent_workflow/rounds/round-N/`.

## Roles

- `reviewer`: first-time external reviewer. It sees only the anonymous PDF,
  judges WWW novelty, significance, Web fit,
  baselines, presentation, and experimental support, and may not edit the paper.
- `modifier`: author-side experimenter and writer. It may change code,
  regenerate figures, rerun experiments, and edit LaTeX to make the strongest
  truthful case, including selective use of relevant exploratory details.
- `coordinator`: checks raw evidence and claim provenance, records artifacts,
  runs compilation/audits, and schedules the next independent review.

## Round protocol

1. Record the manuscript and code snapshot hashes for the round.
2. Run the reviewer against that snapshot and save `review.md` and
   `review.json`.
3. Run the modifier against the review. Save `change_plan.md`, the patch, and
   all generated evidence under the same round.
4. Compile the manuscript and run `scripts/audit_paper_protocol.py`; save the
   logs and updated archive manifest.
5. Re-review the resulting snapshot. Continue while any blocker remains.

## Acceptance gate

The loop stops only when the latest independent review reports no P0/P1 issue,
all headline numbers have a traceable artifact, the protocol audit has no
unresolved leakage or split claims, the PDF compiles without errors, and the
reviewer gives at least weak accept (or an equivalent WWW score). Web-service
claims additionally need valid business transactions, independent sessions,
same-budget fixed and history-only controls, operational outcomes, and measured
end-to-end cost. A high score alone never overrides an evidence or integrity
blocker.

## Evidence rule

Existing results may be used only when the data path and split are documented.
If a requested claim cannot be established from the repository, the modifier
must weaken or remove the claim and mark the missing experiment as future work.
No agent may fabricate runs, hosts, baselines, statistical significance, or
cross-session/generalization claims.

## Claim and result handling

The modifier should state the paper's strongest supported contribution early
and write for reader interest. Limitations belong in the evaluation scope and
limitations discussion; they should not dilute every contribution sentence.
Strong framing must remain evidence-bounded. The main paper must retain the
predefined primary outcomes and matched controls that could change its central
conclusion. Secondary micro-diagnostics may be summarized, moved to the
appendix or artifact, or omitted from the paper when they do not support a
claim; their selection must not change experiment denominators or conceal a
failed workload, a material counterexample, or a negative primary comparison.
The M1 PTS-GPU 0.18% value is
traceable to one online JSON report, but its `run-05` target overlaps the
five-run corpus used to train/select the model grid. Its 15.035 s of selected
probes excludes 7.930 s of context collection (22.965 s total), so it is not
an independent held-out result or a paired speedup measurement. The supplied
M6/T4 value (0.14%) still lacks raw runs and a hardware manifest. Keep these
GPU values separate from the audited M1 cross-validation result. Planned and
missing evidence is tracked in `pending_experiments.csv`.

`workflow.yaml` contains the machine-readable state and artifact contract;
`reviewer_prompt.md` and `modifier_prompt.md` are the role prompts.
