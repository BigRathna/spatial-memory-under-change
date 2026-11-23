"""Diagnostics behind the failure case in REPORT.md.

Sweeps the number of tracked objects and measures current-query success@2 for
each memory, so the report can state where each representation breaks instead
of asserting it. Writes ``results/diagnostics.csv`` and
``results/diagnostics.json``.

Usage::

    .venv/bin/python -m src.diagnostics --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .demo import MEMORY_FACTORIES, MEMORY_ORDER
from .metrics import aggregate, build_schedule, evaluate_memory
from .world import run_episode

OBJECT_COUNTS = (3, 5, 10, 20, 30)
STEPS = 60


def sweep_one(n_objects: int, seeds, n_per_type: int = 10):
    """Return {memory: [per-seed metric dicts]} for one object count."""
    out = {name: [] for name in MEMORY_ORDER}
    for seed in seeds:
        gt, obs_steps, _agents, _world = run_episode(
            seed, steps=STEPS, n_objects=n_objects
        )
        sched = build_schedule(gt, obs_steps, seed, n_per_type=n_per_type)
        for name in MEMORY_ORDER:
            mem = MEMORY_FACTORIES[name](seed)
            out[name].append(evaluate_memory(mem, obs_steps, sched))
    return out


def make_diagnostic_figure(rows, out_path: Path) -> None:
    """Current-query success@2 against the number of tracked objects."""
    colors = {"graph": "tab:blue", "dense": "tab:orange", "implicit": "tab:green"}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name in MEMORY_ORDER:
        sel = [r for r in rows if r["memory"] == name]
        xs = [r["n_objects"] for r in sel]
        ys = [r["current_success@2_mean"] for r in sel]
        es = [r["current_success@2_std"] for r in sel]
        style = {"graph": "--o", "dense": "-o", "implicit": "-o"}[name]
        ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=3, color=colors[name],
                    label=name)
    ax.set_xlabel("objects in the world")
    ax.set_ylabel("current success@2")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Failure case: accuracy vs number of objects")
    ax.legend(fontsize=8, title="graph/dense overlap", title_fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    t0 = time.perf_counter()
    rows = []
    for n_objects in OBJECT_COUNTS:
        per_mem = sweep_one(n_objects, seeds)
        for name in MEMORY_ORDER:
            agg = aggregate(per_mem[name])
            row = {
                "n_objects": n_objects,
                "memory": name,
                "current_success@2_mean": agg["current_success@2"][0],
                "current_success@2_std": agg["current_success@2"][1],
                "past_success@2_mean": agg["past_success@2"][0],
                "past_success@2_std": agg["past_success@2"][1],
                "stale_answer_rate_mean": agg["stale_answer_rate"][0],
                "stale_answer_rate_std": agg["stale_answer_rate"][1],
                "size_bytes_mean": agg["size_bytes"][0],
            }
            rows.append(row)
        print(f"n_objects={n_objects} done in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    fields = list(rows[0])
    with open(out_dir / "diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    payload = {
        "config": {"object_counts": list(OBJECT_COUNTS), "seeds": seeds,
                   "steps": STEPS},
        "wall_seconds": time.perf_counter() - t0,
        "rows": rows,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2))
    make_diagnostic_figure(rows, out_dir / "figure_diagnostics.png")

    print(f"\n{'objects':>8} {'memory':>9} {'current@2':>18} {'stale_answer':>14}")
    for row in rows:
        print(f"{row['n_objects']:8d} {row['memory']:>9} "
              f"{row['current_success@2_mean']:9.3f} +/- "
              f"{row['current_success@2_std']:.3f} "
              f"{row['stale_answer_rate_mean']:11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
