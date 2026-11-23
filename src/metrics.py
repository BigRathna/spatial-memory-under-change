"""Query generation from ground truth and metric computation.

Conventions used throughout:

- World time ``t`` runs over ``1..T``; ``gt[t - 1]`` is the ground-truth
  snapshot at ``t``.
- ``LAG`` = 10 steps for past and region-change queries.
- ``STALE_AGE`` = 15 steps for stale queries.
- ``SUCCESS_DIST`` = 2.0 cells (Euclidean) for success@2.
- Ground-truth stale set at ``t``: every id observed at least once in the
  episode whose last observation is at least ``STALE_AGE`` steps old
  (``t - last_obs >= STALE_AGE``), whether or not the object is still
  present. This matches the memory's point of view: a memory can only ever
  report ids it has seen.
- ``stale_answer_rate``: fraction of *answered* current queries whose
  returned position is more than ``SUCCESS_DIST`` from the object's true
  position at query time, i.e. the memory confidently reports outdated
  information.
- Region change: the symmetric difference of the object sets inside the box
  at ``t`` and at ``t - LAG``, in both prediction and ground truth.
"""

from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .world import GRID_HI, GRID_LO

LAG = 10
STALE_AGE = 15
SUCCESS_DIST = 2.0

Bounds = Tuple[np.ndarray, np.ndarray]


def _inside(pos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.all(pos >= lo)) and bool(np.all(pos <= hi))


def _region_members(gt: Sequence[dict], t: int, bounds: Bounds) -> set:
    lo, hi = bounds
    return {
        oid
        for oid, (pos, _cls) in gt[t - 1].items()
        if _inside(pos, lo, hi)
    }


def _region_change(gt: Sequence[dict], t: int, bounds: Bounds, lag: int) -> set:
    now = _region_members(gt, t, bounds)
    before = _region_members(gt, t - lag, bounds)
    return now ^ before


def build_schedule(
    gt: Sequence[dict],
    obs_steps: Sequence[Sequence],
    seed: int,
    n_per_type: int = 25,
    lag: int = LAG,
    stale_age: int = STALE_AGE,
) -> Dict[str, list]:
    """Draw the query schedule for one seed from ground truth.

    Returns a dict with lists ``current``, ``past``, ``region``, ``stale``
    and the per-step last-observation maps ``last_obs`` used for stale ground
    truth. Entries may be fewer than ``n_per_type`` if the episode is too
    short to support a query type.
    """
    T = len(gt)
    rng = np.random.default_rng(900_000 + seed)

    # last_obs[t] maps id -> latest step <= t at which it was observed.
    last_obs: Dict[int, Dict[int, int]] = {}
    running: Dict[int, int] = {}
    for t in range(1, T + 1):
        for o in obs_steps[t - 1]:
            running[o.oid] = t
        last_obs[t] = dict(running)
    ever = sorted(running)

    current: List[Tuple[int, int, np.ndarray]] = []
    past: List[Tuple[int, int, np.ndarray]] = []
    region: List[Tuple[int, Bounds, set]] = []
    stale: List[Tuple[int, set]] = []

    for _ in range(n_per_type):
        # -- current: where is oid at t?
        for _try in range(100):
            t = int(rng.integers(1, T + 1))
            ids = sorted(gt[t - 1])
            if ids:
                oid = ids[int(rng.integers(len(ids)))]
                current.append((t, oid, gt[t - 1][oid][0].copy()))
                break

        # -- past: where was oid at t - lag?
        if T > lag:
            for _try in range(100):
                t = int(rng.integers(lag + 1, T + 1))
                ids = sorted(gt[t - lag - 1])
                if ids:
                    oid = ids[int(rng.integers(len(ids)))]
                    past.append((t, oid, gt[t - lag - 1][oid][0].copy()))
                    break

        # -- region change: which objects entered or left the box since
        #    t - lag? Prefer samples with a non-empty ground-truth change.
        if T > lag:
            fallback = None
            for _try in range(60):
                t = int(rng.integers(lag + 1, T + 1))
                side = rng.uniform(4.0, 8.0, 3)
                lo = rng.uniform(GRID_LO, GRID_HI - side)
                hi = lo + side
                bounds = (lo.copy(), hi.copy())
                changed = _region_change(gt, t, bounds, lag)
                if changed:
                    fallback = (t, bounds, changed)
                    break
                if fallback is None:
                    fallback = (t, bounds, changed)
            if fallback is not None:
                region.append(fallback)

        # -- stale: which ids have not been observed for stale_age steps?
        if T >= stale_age:
            fallback = None
            for _try in range(100):
                t = int(rng.integers(stale_age, T + 1))
                gt_stale = {oid for oid, s in last_obs[t].items()
                            if t - s >= stale_age}
                if gt_stale:
                    fallback = (t, gt_stale)
                    break
                if fallback is None:
                    fallback = (t, gt_stale)
            if fallback is not None:
                stale.append(fallback)

    return {
        "current": current,
        "past": past,
        "region": region,
        "stale": stale,
        "last_obs": last_obs,
    }


def _prf(pred: set, gt_set: set) -> Tuple[Optional[float], Optional[float]]:
    inter = len(pred & gt_set)
    precision = inter / len(pred) if pred else None
    recall = inter / len(gt_set) if gt_set else None
    return precision, recall


def evaluate_memory(
    mem,
    obs_steps: Sequence[Sequence],
    schedule: Dict[str, list],
    growth_steps: Sequence[int] = (10, 30, 60),
    lag: int = LAG,
    stale_age: int = STALE_AGE,
    latency_checkpoints: Sequence[int] = (),
    latency_window: int = 20,
) -> Dict[str, Optional[float]]:
    """Replay one episode through ``mem`` and score the scheduled queries.

    Queries are evaluated online: the queries scheduled at step ``t`` run
    right after ``update`` at ``t``, so no memory ever sees future data.
    Update and query latency are wall-clock means; peak memory is the
    ``tracemalloc`` peak during the replay.

    When ``latency_checkpoints`` is given, ``query_ms@C`` is the mean query
    latency of queries issued in the window ``(C - latency_window, C]``.
    This separates early-episode latency from late-episode latency in a long
    run; the default (no checkpoints) leaves the metric set unchanged.
    """
    T = len(obs_steps)
    cur_at: Dict[int, list] = defaultdict(list)
    past_at: Dict[int, list] = defaultdict(list)
    reg_at: Dict[int, list] = defaultdict(list)
    stale_at: Dict[int, list] = defaultdict(list)
    for t, oid, true in schedule["current"]:
        cur_at[t].append((oid, true))
    for t, oid, true in schedule["past"]:
        past_at[t].append((oid, true))
    for t, bounds, changed in schedule["region"]:
        reg_at[t].append((bounds, changed))
    for t, gt_stale in schedule["stale"]:
        stale_at[t].append(gt_stale)

    update_ms: List[float] = []
    query_ms: List[float] = []
    query_samples: List[Tuple[int, float]] = []
    cur_hit = cur_ans = 0
    past_hit = past_ans = 0
    stale_answers = 0
    reg_p: List[float] = []
    reg_r: List[float] = []
    st_p: List[float] = []
    st_r: List[float] = []
    stale_total = 0
    stale_answered = 0
    growth: Dict[int, int] = {}

    tracemalloc.start()
    try:
        for t in range(1, T + 1):
            t0 = time.perf_counter()
            mem.update(obs_steps[t - 1], t)
            update_ms.append((time.perf_counter() - t0) * 1e3)

            if t in growth_steps:
                growth[t] = mem.size_bytes()

            for oid, true in cur_at.get(t, ()):
                t0 = time.perf_counter()
                ans = mem.query_current(oid, t)
                dt = (time.perf_counter() - t0) * 1e3
                query_ms.append(dt)
                query_samples.append((t, dt))
                if ans is None:
                    continue
                cur_ans += 1
                d = float(np.linalg.norm(np.asarray(ans[0]) - true))
                if d <= SUCCESS_DIST:
                    cur_hit += 1
                else:
                    stale_answers += 1

            for oid, true in past_at.get(t, ()):
                t0 = time.perf_counter()
                ans = mem.query_at(oid, t - lag)
                dt = (time.perf_counter() - t0) * 1e3
                query_ms.append(dt)
                query_samples.append((t, dt))
                if ans is None:
                    continue
                past_ans += 1
                d = float(np.linalg.norm(np.asarray(ans) - true))
                if d <= SUCCESS_DIST:
                    past_hit += 1

            for bounds, gt_changed in reg_at.get(t, ()):
                t0 = time.perf_counter()
                now = set(mem.query_region(bounds, t))
                prev = set(mem.query_region(bounds, t - lag))
                dt = (time.perf_counter() - t0) * 1e3
                query_ms.append(dt)
                query_samples.append((t, dt))
                pred = now ^ prev
                p, r = _prf(pred, gt_changed)
                if p is not None:
                    reg_p.append(p)
                if r is not None:
                    reg_r.append(r)

            for gt_stale in stale_at.get(t, ()):
                t0 = time.perf_counter()
                pred = set(mem.stale_ids(t, stale_age))
                dt = (time.perf_counter() - t0) * 1e3
                query_ms.append(dt)
                query_samples.append((t, dt))
                stale_total += 1
                stale_answered += int(bool(pred))
                p, r = _prf(pred, gt_stale)
                if p is not None:
                    st_p.append(p)
                if r is not None:
                    st_r.append(r)

        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    n_cur = len(schedule["current"])
    n_past = len(schedule["past"])

    def _mean(xs) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    out: Dict[str, Optional[float]] = {
        "current_success@2": cur_hit / n_cur if n_cur else None,
        "past_success@2": past_hit / n_past if n_past else None,
        "region_precision": _mean(reg_p),
        "region_recall": _mean(reg_r),
        "stale_precision": _mean(st_p),
        "stale_recall": _mean(st_r),
        "stale_coverage": (
            stale_answered / stale_total if stale_total else None
        ),
        "stale_answer_rate": stale_answers / cur_ans if cur_ans else None,
        "size_bytes": float(mem.size_bytes()),
        "peak_mem_bytes": float(peak),
        "update_ms": _mean(update_ms),
        "query_ms": _mean(query_ms),
    }
    for g in growth_steps:
        out[f"growth@{g}"] = float(growth[g]) if g in growth else None
    for cp in latency_checkpoints:
        low = int(cp) - int(latency_window)
        vals = [dt for step, dt in query_samples if low < step <= int(cp)]
        out[f"query_ms@{int(cp)}"] = float(np.mean(vals)) if vals else None
    if hasattr(mem, "collisions"):
        out["voxel_collisions"] = float(mem.collisions)
    return out


METRIC_KEYS = [
    "current_success@2",
    "past_success@2",
    "region_precision",
    "region_recall",
    "stale_precision",
    "stale_recall",
    "stale_answer_rate",
    "size_bytes",
    "peak_mem_bytes",
    "update_ms",
    "query_ms",
    "growth@10",
    "growth@30",
    "growth@60",
    "voxel_collisions",
]


def aggregate(per_seed: List[Dict[str, Optional[float]]]) -> Dict[str, Tuple[float, float]]:
    """Mean and sample standard deviation (ddof=1) of each metric across
    seeds. Metrics that are ``None`` in every seed stay ``None``."""
    out: Dict[str, Tuple[float, float]] = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_seed if m.get(key) is not None]
        if not vals:
            out[key] = None
            continue
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[key] = (float(np.mean(vals)), std)
    return out
