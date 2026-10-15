#!/usr/bin/env python3
"""Cross-fitted probe-input training diagnostic for a PTS suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cv_helpers():
    path = ROOT / "scripts" / "paper_reconstruct_cv_extras.py"
    spec = importlib.util.spec_from_file_location("paper_reconstruct_cv_extras", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CV helpers: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--pts-suite", required=True, choices=("cpu", "pts/nvidia-gpu-compute"))
    parser.add_argument("--probe-dataset", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-duration-s", type=float, default=4.0)
    parser.add_argument("--xi-overhead-s", type=float, default=3.0)
    parser.add_argument("--train-aug", type=int, default=20)
    parser.add_argument("--train-k-min", type=int, default=2)
    parser.add_argument("--train-k-max", type=int, default=6)
    parser.add_argument("--reconstructor", choices=("lightgbm", "xgboost"), default="lightgbm")
    parser.add_argument("--lgbm-estimators", type=int, default=30)
    parser.add_argument("--xgb-estimators", type=int, default=100)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from scripts.run_corrected_oof_unixbench import (
        _build_crossfitted_reconstructor_training_matrix,
        _crossfit_training_probe_predictions,
    )
    from moebench.hybrid.eval import evaluate_single_run_offline
    from moebench.probe.model_train import train_probe_bundle
    from moebench.router.feature_vectorizer import XiVectorizer

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    session_tag = "cpu" if args.pts_suite == "cpu" else "pts_nvidia-gpu-compute"
    rows, test_ids = cv._load_phoronix_rows(
        args.dataset_root.resolve(), f"{args.machine}_{session_tag}_*/run-*.json", args.pts_suite, "logmean"
    )
    if len(rows) < args.folds:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}/{args.pts_suite}")
    args.train_k_max = min(args.train_k_max, len(test_ids))
    args.top_k = min(args.top_k, len(test_ids))
    if args.train_k_min > args.train_k_max:
        raise SystemExit("train-k-min exceeds available PTS profiles")
    probe_path = args.probe_dataset or (args.dataset_root / "models" / args.machine / ("probe_dataset_pts_cpu.json" if args.pts_suite == "cpu" else "probe_dataset_pts_gpu.json"))
    probe_dataset = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe_dataset.get("benchmark") != "phoronix" or str(probe_dataset.get("pts_suite")) != args.pts_suite:
        raise SystemExit("probe dataset benchmark/suite mismatch")
    run_key = lambda value: (Path(value).parent.name, Path(value).name)
    probe_samples = {run_key(s["source_run"]) + (str(s["test_id"]),): s for s in probe_dataset.get("samples", []) if s.get("source_run") and s.get("test_id")}
    rng = np.random.RandomState(args.seed); indices = np.arange(len(rows)); rng.shuffle(indices)
    folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
    xi_vec = XiVectorizer(); reports: list[dict[str, Any] | None] = [None] * len(rows); manifests=[]
    for fold_id, val_indices in enumerate(folds):
        val_set=set(val_indices); train_indices=[i for i in range(len(rows)) if i not in val_set]; train_rows=[rows[i] for i in train_indices]
        train_keys={run_key(r["path"]) for r in train_rows}; outer=dict(probe_dataset); outer["samples"]=[s for s in probe_dataset.get("samples", []) if s.get("source_run") and run_key(s["source_run"]) in train_keys]
        outer_bundle=train_probe_bundle(outer, model_type="lightgbm")
        crossfit, inner=_crossfit_training_probe_predictions(train_rows, probe_dataset, test_ids, seed=args.seed+fold_id*1009)
        x_train,y_train=_build_crossfitted_reconstructor_training_matrix(train_rows,crossfit,probe_samples,xi_vec,test_ids,seed=args.seed+fold_id*997,train_aug=args.train_aug,train_k_min=args.train_k_min,train_k_max=args.train_k_max,benchmark="phoronix",log1p_value=False)
        recon=cv.rte.sklearn_export_bundle(args.reconstructor,x_train,y_train,auto_install=False,lgbm_estimators=args.lgbm_estimators,xgb_estimators=args.xgb_estimators,log1p_partial_index=False,test_ids=test_ids,with_uncertainty=False,benchmark="phoronix"); recon["pts_suite"]=args.pts_suite
        router=cv._train_fold_router(train_rows,test_ids,xi_vec,seed=args.seed+fold_id)
        for i in val_indices:
            reports[i]=evaluate_single_run_offline(ds=rows[i]["ds"],run_path=Path(rows[i]["path"]),router_meta=router,recon_bundle=recon,probe_bundle=outer_bundle,probe_index=probe_samples,top_k=args.top_k,probe_duration_s=args.probe_duration_s,xi_overhead_s=args.xi_overhead_s,benchmark="phoronix",recon_test_ids=test_ids)
        manifests.append({"fold":fold_id,"train_runs":[r["path"] for r in train_rows],"heldout_runs":[rows[i]["path"] for i in val_indices],"reconstructor_train_rows":int(x_train.shape[0]),"reconstructor_probe_inputs":"inner-OOF probe predictions and probe wall times","inner_probe_folds":inner})
        print(f"fold {fold_id+1}/{len(folds)} complete",file=sys.stderr)
    valid=[r for r in reports if r is not None and not r.get("skipped")]; errors=[float(r["comparison"]["suite_relative_error"]) for r in valid]
    report={"schema":"moebench.hybrid.phoronix_crossfit_reconstructor_oof.v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"machine":args.machine,"benchmark":"phoronix","pts_suite":args.pts_suite,"pts_suite_target":"logmean","protocol":"outer run-level OOF with inner-OOF probe-predicted training features; outer held-out inference uses outer-training probe model","folds":args.folds,"seed":args.seed,"top_k":args.top_k,"train_aug":args.train_aug,"train_k_range":[args.train_k_min,args.train_k_max],"reconstructor":args.reconstructor,"reconstructor_estimators":args.lgbm_estimators if args.reconstructor=="lightgbm" else args.xgb_estimators,"input_artifacts":{"probe_dataset":str(probe_path.resolve()),"probe_dataset_sha256":_sha256(probe_path),"runs":[{"path":r["path"],"sha256":_sha256(Path(r["path"]))} for r in rows]},"fold_manifest":manifests,"summary":{"n_runs":len(valid),"mean_relative_error_percent":float(np.mean(errors)*100) if errors else None,"median_relative_error_percent":float(np.median(errors)*100) if errors else None,"p95_relative_error_percent":float(np.percentile(errors,95)*100) if errors else None,"max_relative_error_percent":float(np.max(errors)*100) if errors else None},"per_run":reports}
    args.report_json.parent.mkdir(parents=True,exist_ok=True); args.report_json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(report["summary"],indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
