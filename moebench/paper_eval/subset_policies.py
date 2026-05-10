"""Subset-selection policies for offline partial-benchmark reconstruction experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from moebench.reconstruct.data import _ti_for_test
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs


def _unixbench_wall_s(ds: dict[str, Any], tid: str) -> float | None:
    return _ti_for_test(ds, tid, parallel_key="32")


def stable_seed(parts: tuple[Any, ...]) -> int:
    h = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) % (2**31)


# Classic CPU-ish defaults (subset can be truncated when K < len).
FIXED_CPU_MIX = ["dhry2reg", "whetstone-double", "syscall"]
FIXED_IO_MIX = ["fstime", "fsbuffer", "fsdisk"]


def select_eval_subset(
    policy: str,
    *,
    test_ids: list[str],
    k: int,
    ds: dict[str, Any],
    rng: np.random.RandomState,
    seed_parts: tuple[Any, ...],
    router_meta: dict[str, Any] | None = None,
    profile_wall_seconds: Callable[[dict[str, Any], str], float | None] | None = None,
) -> set[str]:
    """
    Choose exactly ``k`` distinct subtests from ``test_ids``.

    Policies:
    - ``random``: seeded uniform subset (reproducible via seed_parts).
    - ``fixed_first_k``: first ``k`` entries in ``test_ids`` order.
    - ``fixed_cpu_mix``: cycles through FIXED_CPU_MIX then fills from test_ids order.
    - ``fixed_io_mix``: same with FIXED_IO_MIX.
    - ``greedy_slowest``: ``k`` largest wall-clock seconds on this sample (UnixBench: ``ti`` key 32;
      PTS: pass ``profile_wall_seconds``, e.g. ``time_seconds_for_profile``).
    - ``greedy_fastest``: ``k`` smallest positive wall times.
    - ``router``: Top-K from a trained router checkpoint (requires ``router_meta``).

    ``profile_wall_seconds(ds, test_id)`` overrides UnixBench timing for greedy policies.
    """
    k = max(1, min(int(k), len(test_ids)))
    tid_set = list(test_ids)
    wall_fn = profile_wall_seconds or _unixbench_wall_s

    if policy == "random":
        ex_seed = stable_seed(seed_parts + ("eval_subset",))
        ex_rng = np.random.RandomState(ex_seed)
        chosen = ex_rng.choice(tid_set, size=k, replace=False).tolist()
        return set(str(x) for x in chosen)

    if policy == "fixed_first_k":
        return set(tid_set[:k])

    if policy == "fixed_cpu_mix":
        pref = [x for x in FIXED_CPU_MIX if x in tid_set]
        rest = [x for x in tid_set if x not in pref]
        pool = pref + rest
        return set(pool[:k])

    if policy == "fixed_io_mix":
        pref = [x for x in FIXED_IO_MIX if x in tid_set]
        rest = [x for x in tid_set if x not in pref]
        pool = pref + rest
        return set(pool[:k])

    if policy in ("greedy_slowest", "greedy_fastest"):
        scored: list[tuple[float, str]] = []
        for tid in tid_set:
            t = wall_fn(ds, tid)
            if t is None or t <= 0:
                continue
            scored.append((float(t), tid))
        if len(scored) < k:
            # fallback: fill with random among remaining
            have = {x[1] for x in scored}
            rest = [t for t in tid_set if t not in have]
            ex_seed = stable_seed(seed_parts + ("greedy_fallback",))
            ex_rng = np.random.RandomState(ex_seed)
            extra = ex_rng.choice(rest, size=min(k - len(scored), len(rest)), replace=False).tolist()
            scored.extend((1e-9, str(x)) for x in extra)
        scored.sort(key=lambda x: x[0], reverse=(policy == "greedy_slowest"))
        return {tid for _, tid in scored[:k]}

    if policy == "router":
        if router_meta is None:
            raise ValueError("policy=router requires router_meta")
        xi = ds.get("xi") or {}
        _scores, probs, _eids, etids = predict_expert_scores(router_meta, xi)
        # Map expert_test_ids order to probs
        _, chosen_tests = select_top_k_from_probs(probs, router_meta["expert_ids"], etids, k)
        out = {t for t in chosen_tests if t in tid_set}
        if len(out) < k:
            need = k - len(out)
            rest = [t for t in tid_set if t not in out]
            ex_seed = stable_seed(seed_parts + ("router_fill",))
            ex_rng = np.random.RandomState(ex_seed)
            fill = ex_rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()
            out.update(str(x) for x in fill)
        return out

    raise ValueError(
        f"Unknown policy {policy!r}; try: random, fixed_first_k, fixed_cpu_mix, fixed_io_mix, "
        "greedy_slowest, greedy_fastest, router"
    )
