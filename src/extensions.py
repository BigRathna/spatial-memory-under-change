"""Extensions to the spatial-memory benchmark (experiments A, B and C).

Usage::

    .venv/bin/python -m src.extensions --seeds 5

Experiment A varies the coordinate-MLP capacity (width in 32/64/128, depth
in 1/2) at 3 and 30 objects, with the original online budget.

Experiment B adds four relational and semantic query types. Queries are
drawn from ground truth; each memory answers from the positions it stored.
Class labels for ``nearest_of_class`` are supplied by the query, so the
measurement isolates position retrieval from semantic labelling.

Experiment C adds dense-grid eviction horizons (10 and 20 steps) and runs
the original three memories for 300 steps, recording size and query latency
at 60, 120 and 300 steps.

All outputs go to ``results/extensions/``. The original ``results`` files
are never written. No new dependencies beyond numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .memories import DenseGridMemory, ImplicitMemory, ObjectGraphMemory  # noqa: E402
from .metrics import (  # noqa: E402
    LAG,
    STALE_AGE,
    build_schedule,
    evaluate_memory,
)
from .world import GRID, GRID_HI, GRID_LO, run_episode  # noqa: E402

OUT_DIR = Path("results/extensions")
STEPS = 60

CAPACITY_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (32, 1), (64, 1), (128, 1),
    (32, 2), (64, 2), (128, 2),
)
CAPACITY_OBJECTS = (3, 30)
CAPACITY_KEYS = (
    "current_success@2",
    "past_success@2",
    "stale_answer_rate",
    "update_ms",
    "peak_mem_bytes",
    "size_bytes",
)

REL_TYPES = ("left_of", "right_of", "near", "nearest_of_class")
REL_RADIUS = 3.0

EVICTION_VARIANTS: Tuple[Tuple[str, Optional[int]], ...] = (
    ("no_eviction", None),
    ("evict10", 10),
    ("evict20", 20),
)
EVICTION_KEYS = (
    "current_success@2",
    "past_success@2",
    "stale_precision",
    "stale_recall",
    "stale_coverage",
    "size_bytes",
    "voxel_collisions",
)
HORIZON_CHECKS = (60, 120, 300)
HORIZON_KEYS = (
    "size_bytes",
    "query_ms",
    "current_success@2",
    "past_success@2",
)


# ---------------------------------------------------------------------------
# Evicting dense grid
# ---------------------------------------------------------------------------


class EvictingDenseGridMemory(DenseGridMemory):
    """Dense grid that clears voxels older than ``max_age`` steps.

    Eviction runs at the start of ``update`` so the observations of the
    current step are always written. A voxel whose last write is exactly
    ``max_age`` steps old survives; it is dropped on the next step. The
    arrays keep their shape, so persistent size is unchanged. ``collisions``
    keeps counting writes into an occupied voxel, but cleared voxels no
    longer produce collisions.
    """

    def __init__(self, max_age: int, grid: Tuple[int, int, int] = GRID) -> None:
        if int(max_age) <= 0:
            raise ValueError("max_age must be positive")
        super().__init__(grid)
        self.max_age = int(max_age)
        self.evicted_voxels = 0

    def _evict(self, t: int) -> int:
        cutoff = int(t) - self.max_age
        mask = (self.lastseen >= 0) & (self.lastseen < cutoff)
        n = int(mask.sum())
        if n:
            self.ids[mask] = -1
            self.lastseen[mask] = -1
            self.evicted_voxels += n
        return n

    def update(self, observations: Iterable, t: int) -> None:
        self._evict(int(t))
        super().update(observations, t)


# ---------------------------------------------------------------------------
# Relational query semantics
# ---------------------------------------------------------------------------


def answer_left_of(pos_a, pos_b) -> Optional[bool]:
    if pos_a is None or pos_b is None:
        return None
    return bool(float(np.asarray(pos_a)[0]) < float(np.asarray(pos_b)[0]))


def answer_right_of(pos_a, pos_b) -> Optional[bool]:
    if pos_a is None or pos_b is None:
        return None
    return bool(float(np.asarray(pos_a)[0]) > float(np.asarray(pos_b)[0]))


def answer_near(pos_a, pos_b, radius: float = REL_RADIUS) -> Optional[bool]:
    if pos_a is None or pos_b is None:
        return None
    d = float(np.linalg.norm(np.asarray(pos_a, float) - np.asarray(pos_b, float)))
    return bool(d <= float(radius))


def answer_nearest_of_class(
    positions: Dict[int, np.ndarray],
    class_by_id: Dict[int, int],
    cls: int,
    point,
) -> Optional[int]:
    """Nearest object of ``cls`` to ``point`` among the available positions.

    Ties break on the lower id. Missing positions are simply unavailable.
    """
    target = np.asarray(point, dtype=float)
    best = None
    best_d = np.inf
    for oid, pos in positions.items():
        if int(class_by_id.get(int(oid), -1)) != int(cls):
            continue
        d = float(np.linalg.norm(np.asarray(pos, float) - target))
        if d < best_d or (d == best_d and (best is None or int(oid) < best)):
            best = int(oid)
            best_d = d
    return best


def _pair_truth(gt_t: dict, a: int, b: int, kind: str) -> bool:
    pa = gt_t[a][0]
    pb = gt_t[b][0]
    if kind == "left_of":
        return bool(pa[0] < pb[0])
    if kind == "right_of":
        return bool(pa[0] > pb[0])
    if kind == "near":
        return bool(np.linalg.norm(pa - pb) <= REL_RADIUS)
    raise ValueError(kind)


def _balanced_pairs(gt, seed: int, kind: str, n: int) -> List[dict]:
    """Draw ``n`` pairs with a balanced true/false split where possible."""
    rng = np.random.default_rng(700_000 + seed + REL_TYPES.index(kind))
    T = len(gt)
    target_pos = n // 2
    target_neg = n - target_pos
    positives: List[dict] = []
    negatives: List[dict] = []
    seen = set()
    attempts = 0
    while (
        len(positives) < target_pos or len(negatives) < target_neg
    ) and attempts < 20_000:
        attempts += 1
        t = int(rng.integers(1, T + 1))
        ids = sorted(gt[t - 1])
        if len(ids) < 2:
            continue
        i, j = rng.choice(len(ids), size=2, replace=False)
        a, b = int(ids[int(i)]), int(ids[int(j)])
        if (t, a, b) in seen:
            continue
        truth = _pair_truth(gt[t - 1], a, b, kind)
        q = {"t": t, "a": a, "b": b, "truth": truth}
        if truth and len(positives) < target_pos:
            positives.append(q)
            seen.add((t, a, b))
        elif not truth and len(negatives) < target_neg:
            negatives.append(q)
            seen.add((t, a, b))
    return positives + negatives


def _nearest_queries(gt, seed: int, n: int) -> List[dict]:
    rng = np.random.default_rng(710_000 + seed)
    T = len(gt)
    out: List[dict] = []
    for _ in range(n):
        t = int(rng.integers(1, T + 1))
        rows = gt[t - 1]
        if not rows:
            continue
        classes = sorted({int(cls) for _pos, cls in rows.values()})
        cls = int(classes[int(rng.integers(len(classes)))])
        members = sorted(oid for oid, (_pos, c) in rows.items() if int(c) == cls)
        point = rng.uniform(GRID_LO, GRID_HI)
        truth = min(
            members,
            key=lambda oid: float(np.linalg.norm(rows[oid][0] - point)),
        )
        out.append(
            {
                "t": t,
                "cls": cls,
                "point": point.copy(),
                "truth": int(truth),
                "candidates": [
                    (int(oid), int(c)) for oid, (_pos, c) in rows.items()
                ],
            }
        )
    return out


def build_relational_schedule(gt, seed: int, n_per_type: int = 25) -> Dict[str, list]:
    """Queries for the four relational types, drawn from ground truth."""
    schedule: Dict[str, list] = {"seed": int(seed)}
    for kind in ("left_of", "right_of", "near"):
        schedule[kind] = _balanced_pairs(gt, seed, kind, n_per_type)
    schedule["nearest_of_class"] = _nearest_queries(gt, seed, n_per_type)
    return schedule


def _memory_pos(mem, oid: int, t: int):
    ans = mem.query_current(int(oid), int(t))
    return None if ans is None else np.asarray(ans[0], dtype=float)


def evaluate_relational(mem, obs_steps: Sequence, schedule: Dict[str, list]) -> Dict[str, float]:
    """Replay an episode and score the relational queries online."""
    by_t: Dict[int, List[Tuple[str, dict]]] = defaultdict(list)
    for kind in REL_TYPES:
        for q in schedule.get(kind, ()):
            by_t[int(q["t"])].append((kind, q))

    hits = {kind: 0 for kind in REL_TYPES}
    counts = {kind: 0 for kind in REL_TYPES}
    for t in range(1, len(obs_steps) + 1):
        mem.update(obs_steps[t - 1], t)
        for kind, q in by_t.get(t, ()):
            if kind == "left_of":
                pred = answer_left_of(_memory_pos(mem, q["a"], t),
                                      _memory_pos(mem, q["b"], t))
            elif kind == "right_of":
                pred = answer_right_of(_memory_pos(mem, q["a"], t),
                                       _memory_pos(mem, q["b"], t))
            elif kind == "near":
                pred = answer_near(_memory_pos(mem, q["a"], t),
                                   _memory_pos(mem, q["b"], t))
            else:
                positions = {}
                for oid, cls in q["candidates"]:
                    if int(cls) != int(q["cls"]):
                        continue
                    pos = _memory_pos(mem, int(oid), t)
                    if pos is not None:
                        positions[int(oid)] = pos
                pred = answer_nearest_of_class(
                    positions,
                    {oid: int(q["cls"]) for oid in positions},
                    int(q["cls"]),
                    q["point"],
                )
            counts[kind] += 1
            hits[kind] += int(pred is not None and pred == q["truth"])

    out: Dict[str, float] = {}
    for kind in REL_TYPES:
        out[f"{kind}_accuracy"] = (
            hits[kind] / counts[kind] if counts[kind] else None
        )
        out[f"{kind}_n"] = counts[kind]
    return out


# ---------------------------------------------------------------------------
# Aggregation and output helpers
# ---------------------------------------------------------------------------


def _aggregate(per_seed: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for key in keys:
        vals = [d[key] for d in per_seed if d.get(key) is not None]
        if not vals:
            out[key] = (None, None)
        elif len(vals) == 1:
            out[key] = (float(vals[0]), 0.0)
        else:
            out[key] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)),
            )
    return out


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _factory(name: str, seed: int):
    if name == "graph":
        return ObjectGraphMemory()
    if name == "dense":
        return DenseGridMemory()
    if name == "implicit":
        return ImplicitMemory(seed=seed)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Experiment A: capacity
# ---------------------------------------------------------------------------


def run_capacity(seeds: Sequence[int], n_per_type: int = 25) -> dict:
    per_seed: Dict[str, List[dict]] = {}
    rows: List[dict] = []
    for n_objects in CAPACITY_OBJECTS:
        for seed in seeds:
            gt, obs_steps, _agents, _world = run_episode(
                seed, steps=STEPS, n_objects=n_objects
            )
            sched = build_schedule(gt, obs_steps, seed, n_per_type=n_per_type)
            for hidden, depth in CAPACITY_CONFIGS:
                name = f"implicit_w{hidden}_d{depth}"
                mem = ImplicitMemory(seed=seed, hidden=hidden, depth=depth)
                res = evaluate_memory(mem, obs_steps, sched)
                per_seed.setdefault(f"{n_objects}:{name}", []).append(res)
            for name in ("graph", "dense"):
                mem = _factory(name, seed)
                res = evaluate_memory(mem, obs_steps, sched)
                per_seed.setdefault(f"{n_objects}:{name}", []).append(res)

    for n_objects in CAPACITY_OBJECTS:
        for name in [f"implicit_w{h}_d{d}" for h, d in CAPACITY_CONFIGS] + [
            "graph", "dense"
        ]:
            agg = _aggregate(per_seed[f"{n_objects}:{name}"], CAPACITY_KEYS)
            row = {
                "n_objects": n_objects,
                "memory": name,
                "width": (
                    int(name.split("_w")[1].split("_")[0])
                    if name.startswith("implicit") else ""
                ),
                "depth": (
                    int(name.rsplit("_d", 1)[1]) if name.startswith("implicit") else ""
                ),
                "n_seeds": len(seeds),
            }
            for key in CAPACITY_KEYS:
                mean, std = agg[key]
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
            rows.append(row)
    return {
        "rows": rows,
        "payload": {
            "config": {
                "seeds": list(seeds),
                "steps": STEPS,
                "queries_per_type": n_per_type,
                "object_counts": list(CAPACITY_OBJECTS),
                "configs": [
                    {"width": w, "depth": d} for w, d in CAPACITY_CONFIGS
                ],
            },
            "per_seed": per_seed,
            "aggregate": rows,
        },
    }


def make_capacity_figure(rows: Sequence[dict], out_path: Path) -> None:
    colors = {32: "tab:blue", 64: "tab:orange", 128: "tab:green"}
    styles = {1: "-", 2: "--"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, ylabel in (
        (axes[0], "current_success@2", "current success@2"),
        (axes[1], "stale_answer_rate", "stale-answer rate"),
    ):
        for hidden, depth in CAPACITY_CONFIGS:
            name = f"implicit_w{hidden}_d{depth}"
            sel = [r for r in rows if r["memory"] == name]
            xs = [r["n_objects"] for r in sel]
            ys = [r[f"{key}_mean"] for r in sel]
            ax.plot(xs, ys, marker="o", color=colors[hidden],
                    linestyle=styles[depth], label=f"w{hidden} d{depth}")
        for name, color, marker in (
            ("graph", "black", "s"),
            ("dense", "gray", "D"),
        ):
            sel = [r for r in rows if r["memory"] == name]
            if not sel:
                continue
            ax.plot([r["n_objects"] for r in sel],
                    [r[f"{key}_mean"] for r in sel],
                    marker=marker, color=color, linestyle=":",
                    label=name, alpha=0.7)
        ax.set_xticks(list(CAPACITY_OBJECTS))
        ax.set_xticklabels([str(v) for v in CAPACITY_OBJECTS])
        ax.set_xlim(0, 33)
        ax.set_xlabel("objects in the world")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_title("Capacity sweep: current success@2")
    axes[1].set_title("Capacity sweep: stale-answer rate")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment B: relational queries
# ---------------------------------------------------------------------------


def run_relational(seeds: Sequence[int], n_per_type: int = 25) -> dict:
    per_seed: Dict[str, List[dict]] = {
        name: [] for name in ("graph", "dense", "implicit")
    }
    chance: List[dict] = []
    for seed in seeds:
        gt, obs_steps, _agents, _world = run_episode(seed, steps=STEPS)
        sched = build_relational_schedule(gt, seed, n_per_type=n_per_type)
        member_counts = []
        for q in sched["nearest_of_class"]:
            classes = defaultdict(int)
            for _oid, cls in q["candidates"]:
                classes[int(cls)] += 1
            member_counts.append(classes[int(q["cls"])])
        chance.append(
            {
                "seed": seed,
                "boolean": 0.5,
                "nearest_of_class": (
                    float(np.mean([1.0 / c for c in member_counts]))
                    if member_counts else None
                ),
            }
        )
        for name in ("graph", "dense", "implicit"):
            mem = _factory(name, seed)
            per_seed[name].append(evaluate_relational(mem, obs_steps, sched))

    rows: List[dict] = []
    for name in ("graph", "dense", "implicit"):
        agg = _aggregate(
            per_seed[name],
            [f"{kind}_accuracy" for kind in REL_TYPES],
        )
        row = {"memory": name, "seeds": len(seeds)}
        for kind in REL_TYPES:
            mean, std = agg[f"{kind}_accuracy"]
            row[f"{kind}_mean"] = mean
            row[f"{kind}_std"] = std
        rows.append(row)
    return {
        "rows": rows,
        "payload": {
            "config": {
                "seeds": list(seeds),
                "steps": STEPS,
                "queries_per_type": n_per_type,
                "radius": REL_RADIUS,
                "class_labels": "ground truth (oracle) for query candidates",
            },
            "chance": chance,
            "per_seed": per_seed,
            "aggregate": rows,
        },
    }


# ---------------------------------------------------------------------------
# Experiment C: maintenance and long horizon
# ---------------------------------------------------------------------------


def run_eviction(seeds: Sequence[int], n_per_type: int = 25) -> dict:
    per_seed: Dict[str, List[dict]] = {
        variant: [] for variant, _age in EVICTION_VARIANTS
    }
    for seed in seeds:
        gt, obs_steps, _agents, _world = run_episode(seed, steps=STEPS)
        sched = build_schedule(gt, obs_steps, seed, n_per_type=n_per_type)
        for variant, max_age in EVICTION_VARIANTS:
            mem = (
                DenseGridMemory() if max_age is None
                else EvictingDenseGridMemory(max_age=int(max_age))
            )
            res = evaluate_memory(mem, obs_steps, sched)
            res["evicted_voxels"] = float(
                getattr(mem, "evicted_voxels", 0)
            )
            per_seed[variant].append(res)

    rows: List[dict] = []
    for variant, max_age in EVICTION_VARIANTS:
        keys = EVICTION_KEYS + ("evicted_voxels",)
        agg = _aggregate(per_seed[variant], keys)
        for key in keys:
            mean, std = agg[key]
            rows.append(
                {
                    "section": "eviction",
                    "config": variant,
                    "max_age": "" if max_age is None else int(max_age),
                    "memory": "dense",
                    "step": "",
                    "metric": key,
                    "mean": mean,
                    "std": std,
                    "n_seeds": len(seeds),
                }
            )
    return {
        "rows": rows,
        "payload": {
            "config": {
                "seeds": list(seeds),
                "steps": STEPS,
                "queries_per_type": n_per_type,
                "variants": [
                    {"name": name, "max_age": age}
                    for name, age in EVICTION_VARIANTS
                ],
            },
            "per_seed": per_seed,
            "aggregate": rows,
        },
    }


def run_horizon(
    seeds: Sequence[int],
    steps: int = 300,
    n_per_type: int = 100,
    window: int = 20,
) -> dict:
    per_seed: Dict[str, List[dict]] = {
        name: [] for name in ("graph", "dense", "implicit")
    }
    for seed in seeds:
        gt, obs_steps, _agents, _world = run_episode(seed, steps=steps)
        sched = build_schedule(gt, obs_steps, seed, n_per_type=n_per_type)
        for name in ("graph", "dense", "implicit"):
            mem = _factory(name, seed)
            res = evaluate_memory(
                mem,
                obs_steps,
                sched,
                growth_steps=HORIZON_CHECKS,
                latency_checkpoints=HORIZON_CHECKS,
                latency_window=window,
            )
            per_seed[name].append(res)

    keys = tuple(
        [f"growth@{step}" for step in HORIZON_CHECKS]
        + [f"query_ms@{step}" for step in HORIZON_CHECKS]
        + list(HORIZON_KEYS)
    )
    rows: List[dict] = []
    for name in ("graph", "dense", "implicit"):
        agg = _aggregate(per_seed[name], keys)
        for step in HORIZON_CHECKS:
            pairs = (
                ("size_bytes", f"growth@{step}"),
                ("query_ms", f"query_ms@{step}"),
                ("current_success@2", "current_success@2"),
                ("past_success@2", "past_success@2"),
            )
            for metric, key in pairs:
                mean, std = agg[key]
                rows.append(
                    {
                        "section": "horizon",
                        "config": "",
                        "max_age": "",
                        "memory": name,
                        "step": step,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n_seeds": len(seeds),
                    }
                )
    return {
        "rows": rows,
        "payload": {
            "config": {
                "seeds": list(seeds),
                "steps": steps,
                "queries_per_type": n_per_type,
                "latency_window": window,
                "checkpoints": list(HORIZON_CHECKS),
            },
            "per_seed": per_seed,
            "aggregate": rows,
        },
    }


def _lookup(rows: Sequence[dict], section: str, metric: str,
            config: str = "", memory: str = "", step=None):
    for r in rows:
        if r["section"] != section:
            continue
        if r["metric"] != metric:
            continue
        if config and r["config"] != config:
            continue
        if memory and r["memory"] != memory:
            continue
        if step is not None and r["step"] != step:
            continue
        return r["mean"], r["std"]
    return None, None


def make_maintenance_figure(
    eviction_rows: Sequence[dict], horizon_rows: Sequence[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    names = [name for name, _age in EVICTION_VARIANTS]
    x = np.arange(len(names))
    width = 0.26
    for k, (metric, label, color) in enumerate(
        (
            ("past_success@2", "past success@2", "tab:blue"),
            ("stale_recall", "stale recall", "tab:orange"),
            ("stale_coverage", "stale coverage", "tab:green"),
        )
    ):
        means = [_lookup(eviction_rows, "eviction", metric, config=n)[0]
                 for n in names]
        vals = [np.nan if v is None else v for v in means]
        ax.bar(x + (k - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Dense-grid eviction")
    ax.set_ylabel("score")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    colors = {"graph": "tab:blue", "dense": "tab:orange", "implicit": "tab:green"}
    for ax, metric, ylabel in (
        (axes[1], "size_bytes", "persistent size [kB]"),
        (axes[2], "query_ms", "query latency [ms]"),
    ):
        for name in ("graph", "dense", "implicit"):
            vals = [_lookup(horizon_rows, "horizon", metric, memory=name,
                            step=step)[0] for step in HORIZON_CHECKS]
            y = [np.nan if v is None else v for v in vals]
            if metric == "size_bytes":
                y = [v / 1000.0 if not np.isnan(v) else v for v in y]
            ax.plot(HORIZON_CHECKS, y, marker="o", color=colors[name],
                    label=name)
        ax.set_xlabel("episode step")
        ax.set_ylabel(ylabel)
        ax.set_title("Long episode: " + ylabel.split(" [")[0])
        ax.set_xticks(list(HORIZON_CHECKS))
        ax.grid(alpha=0.3)
        if metric == "size_bytes":
            ax.set_yscale("log")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds (seed values 0..n-1)")
    parser.add_argument("--queries-per-type", type=int, default=25,
                        help="queries per type in experiments A, B and C")
    parser.add_argument("--horizon-queries", type=int, default=100,
                        help="queries per type in the 300-step episode")
    parser.add_argument("--only", choices=["a", "b", "c", "all"],
                        default="all")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    runtimes: Dict[str, float] = {}

    t_start = time.perf_counter()
    if args.only in ("a", "all"):
        t0 = time.perf_counter()
        cap = run_capacity(seeds, n_per_type=args.queries_per_type)
        write_csv(out_dir / "capacity.csv", cap["rows"])
        cap["payload"]["wall_seconds"] = time.perf_counter() - t0
        write_json(out_dir / "capacity.json", cap["payload"])
        make_capacity_figure(cap["rows"], out_dir / "capacity.png")
        runtimes["capacity"] = time.perf_counter() - t0
        print(f"experiment A done in {runtimes['capacity']:.1f}s", flush=True)

    if args.only in ("b", "all"):
        t0 = time.perf_counter()
        rel = run_relational(seeds, n_per_type=args.queries_per_type)
        write_csv(out_dir / "relational.csv", rel["rows"])
        rel["payload"]["wall_seconds"] = time.perf_counter() - t0
        write_json(out_dir / "relational.json", rel["payload"])
        runtimes["relational"] = time.perf_counter() - t0
        print(f"experiment B done in {runtimes['relational']:.1f}s", flush=True)

    if args.only in ("c", "all"):
        t0 = time.perf_counter()
        evict = run_eviction(seeds, n_per_type=args.queries_per_type)
        t1 = time.perf_counter()
        horizon = run_horizon(seeds, n_per_type=args.horizon_queries)
        t2 = time.perf_counter()
        rows = list(evict["rows"]) + list(horizon["rows"])
        write_csv(out_dir / "maintenance.csv", rows)
        payload = {
            "config": {
                "seeds": seeds,
                "eviction": evict["payload"]["config"],
                "horizon": horizon["payload"]["config"],
            },
            "wall_seconds": t2 - t0,
            "eviction": evict["payload"],
            "horizon": horizon["payload"],
        }
        write_json(out_dir / "maintenance.json", payload)
        make_maintenance_figure(
            evict["rows"], horizon["rows"], out_dir / "maintenance.png"
        )
        runtimes["eviction"] = t1 - t0
        runtimes["horizon"] = t2 - t1
        runtimes["maintenance"] = t2 - t0
        print(
            f"experiment C done in {runtimes['maintenance']:.1f}s "
            f"(eviction {runtimes['eviction']:.1f}s, "
            f"horizon {runtimes['horizon']:.1f}s)",
            flush=True,
        )

    total = time.perf_counter() - t_start
    print(f"\nwall time: {total:.1f}s   outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
