"""Benchmark driver.

Usage::

    .venv/bin/python -m src.demo --seeds 3

Simulates one episode per seed, replays it through the three memories,
evaluates 100 ground-truth queries per seed, and writes
``results/metrics.csv``, ``results/metrics.json``, ``results/table.md`` and
``results/figure.png``. Everything is seeded; only wall-clock latency and
tracemalloc peaks vary between runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .memories import DenseGridMemory, ImplicitMemory, ObjectGraphMemory  # noqa: E402
from .metrics import METRIC_KEYS, aggregate, build_schedule, evaluate_memory  # noqa: E402
from .world import run_episode  # noqa: E402

STEPS = 60
N_PER_TYPE = 25
GROWTH_STEPS = (10, 30, 60)

MEMORY_FACTORIES = {
    "graph": lambda seed: ObjectGraphMemory(),
    "dense": lambda seed: DenseGridMemory(),
    "implicit": lambda seed: ImplicitMemory(seed=seed),
}
MEMORY_ORDER = list(MEMORY_FACTORIES)


def run_seed(seed: int, steps: int = STEPS, n_per_type: int = N_PER_TYPE):
    """One seed: simulate, build the query schedule, score each memory."""
    gt, obs_steps, _agents, world = run_episode(seed, steps=steps)
    schedule = build_schedule(gt, obs_steps, seed, n_per_type=n_per_type)
    results = {}
    for name in MEMORY_ORDER:
        mem = MEMORY_FACTORIES[name](seed)
        results[name] = evaluate_memory(
            mem, obs_steps, schedule, growth_steps=GROWTH_STEPS
        )
    return results


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, tuple):
        mean, std = x
        if mean is None:
            return "-"
        return f"{mean:.3f} +/- {std:.3f}"
    return f"{x:.3f}"


def format_markdown_table(agg_by_memory) -> str:
    """One results table: rows are metrics, columns are memories."""
    header = "| metric | " + " | ".join(MEMORY_ORDER) + " |"
    sep = "|---" * (len(MEMORY_ORDER) + 1) + "|"
    lines = [header, sep]
    for key in METRIC_KEYS:
        row = [key]
        for name in MEMORY_ORDER:
            row.append(_fmt(agg_by_memory[name].get(key)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_figure(agg_by_memory, out_path: Path) -> None:
    """Success@2 vs persistent size, and persistent-size growth over steps."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"graph": "tab:blue", "dense": "tab:orange", "implicit": "tab:green"}

    ax = axes[0]
    for name in MEMORY_ORDER:
        a = agg_by_memory[name]
        size_kb = a["size_bytes"][0] / 1000.0
        cur = a["current_success@2"][0]
        past = a["past_success@2"][0]
        ax.scatter([size_kb], [cur], color=colors[name], marker="o", s=70,
                   zorder=3)
        ax.scatter([size_kb], [past], color=colors[name], marker="s", s=70,
                   facecolors="none", zorder=3)
        ax.annotate(name, (size_kb, cur), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=colors[name])
    ax.scatter([], [], color="gray", marker="o", label="current")
    ax.scatter([], [], color="gray", marker="s", facecolors="none", label="past")
    ax.set_xlim(0, 27)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("persistent size after episode [kB]")
    ax.set_ylabel("success@2")
    ax.set_title("Query accuracy vs persistent size (mean of 3 seeds)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    steps = list(GROWTH_STEPS)
    for name in MEMORY_ORDER:
        a = agg_by_memory[name]
        sizes = [a[f"growth@{s}"][0] / 1000.0 for s in steps]
        ax.plot(steps, sizes, marker="o", color=colors[name], label=name)
    ax.set_xlabel("episode step")
    ax.set_ylabel("persistent size [kB]")
    ax.set_title("Memory growth during the episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3,
                        help="number of seeds (seed values 0..n-1)")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--queries-per-type", type=int, default=N_PER_TYPE)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    per_seed = {}
    for seed in range(args.seeds):
        t0 = time.perf_counter()
        per_seed[seed] = run_seed(
            seed, steps=args.steps, n_per_type=args.queries_per_type
        )
        print(f"seed {seed} done in {time.perf_counter() - t0:.1f}s",
              file=sys.stderr)

    agg = {name: aggregate([per_seed[s][name] for s in per_seed])
           for name in MEMORY_ORDER}
    wall = time.perf_counter() - t_start

    payload = {
        "config": {
            "seeds": list(per_seed),
            "steps": args.steps,
            "queries_per_type": args.queries_per_type,
            "growth_steps": list(GROWTH_STEPS),
            "python": sys.version.split()[0],
        },
        "wall_seconds": wall,
        "per_seed": {str(s): per_seed[s] for s in per_seed},
        "aggregate": agg,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["memory", "metric", "mean", "std"])
        for name in MEMORY_ORDER:
            for key in METRIC_KEYS:
                val = agg[name].get(key)
                if val is None:
                    writer.writerow([name, key, "", ""])
                else:
                    writer.writerow([name, key, f"{val[0]:.6f}", f"{val[1]:.6f}"])

    table = format_markdown_table(agg)
    (out_dir / "table.md").write_text(table + "\n")

    make_figure(agg, out_dir / "figure.png")

    print(table)
    print(f"\nwall time: {wall:.1f}s   results written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
