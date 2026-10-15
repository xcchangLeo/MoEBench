# Suite Prediction Sensitivity Audit

## Scope

This is a focused diagnostic for M5 (`iZbp1acaw5wdllhz47922rZ`), randomized fold 3 (zero-based fold 2), held-out run `run-03.json`. The router, probe model, and reconstructor were retrained with the policy OOF configuration: seed 42, five folds, 20 augmented rows per training run, and a 200-estimator LightGBM multi-output reconstructor. This diagnostic does not change manuscript results.

## Findings

- The reconstruction training matrix had 80 rows and 100 features: 64 xi features plus 36 probe mask/index/time features for 12 UnixBench tests. The suite target varied in training (`std = 38.7807`), so the output was not constant because all training labels were identical.
- The separately fitted suite-output LightGBM contained 200 trees and 600 split nodes. All splits used only five xi features: `proc_cpu_utilization_ratio`, `vm_page_faults_per_sec`, `vm_minor_faults_per_sec`, `loadavg_1m`, and `loadavg_5m`. Every probe mask/index/time feature had zero split count, zero gain, and zero feature importance for the suite output.
- Rebuilding the held-out feature row for each policy reproduced the recorded suite prediction exactly (`1640.3910506772634`), while the four policies had different selected IDs and probe predictions.
- For each policy, increasing one selected probe score by 10% and its probe time by 10% changed the suite prediction by exactly zero. Thus the model's suite head is insensitive to probe features in this fold; the matching outputs are not caused by report overwrite or fixed-policy selection reuse.

## Interpretation

This representative fold identifies trained-model feature insensitivity as the direct cause of identical policy suite predictions. The inference path does construct policy-specific partial rows, but this suite estimator learned only xi splits. The model class fits an independent estimator per output, and suite targets are repeated across augmented partial rows; consequently, the suite head can favor context-only splits even when the component heads use probe information. This is an empirical diagnosis for one fold, not proof that all host/fold suite heads ignore probes. The policy comparison still shows no learned-selection accuracy advantage under the current protocol.

## Reproduction References

- Policy implementation: `scripts/run_hybrid_oof_unixbench_policies.py`, `_evaluate_policy_run`.
- Training rows and output estimator: `scripts/reconstruct_train_eval.py`, `build_augmented_train_matrix`, `fit_sklearn_multioutput`, and `sklearn_export_bundle`.
- Partial feature construction and final suite output: `moebench/reconstruct/data.py`, `build_partial_feature_row_from_executed_tests`; `moebench/reconstruct/inference.py`, `predict_from_partial`.
- Audited policy reports: `iZbp1acaw5wdllhz47922rZ__{learned,fixed_cpu_mix,fixed_io_mix,fixed_first_k}.json`.
