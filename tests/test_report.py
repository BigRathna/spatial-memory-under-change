"""Check that REPORT.md's numbers come from the result files.

This is the guard for the "no hand-typed numbers" rule: every cell in the
report's results table and in the failure-case table must match a value that
the benchmark wrote to ``results/``.
"""

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "REPORT.md"
METRICS = ROOT / "results" / "metrics.csv"
DIAGNOSTICS = ROOT / "results" / "diagnostics.csv"

# Report row label -> metric key in results/metrics.csv.
ROW_TO_KEY = {
    "current success@2": "current_success@2",
    "past success@2": "past_success@2",
    "region precision": "region_precision",
    "region recall": "region_recall",
    "stale precision": "stale_precision",
    "stale recall": "stale_recall",
    "stale-answer rate": "stale_answer_rate",
    "persistent size [B]": "size_bytes",
    "peak memory [B]": "peak_mem_bytes",
    "update latency [ms]": "update_ms",
    "query latency [ms]": "query_ms",
    "size at step 10 [B]": "growth@10",
    "size at step 30 [B]": "growth@30",
    "size at step 60 [B]": "growth@60",
    "voxel collisions": "voxel_collisions",
}

COL_TO_MEMORY = {"ObjectGraph": "graph", "DenseGrid": "dense", "Implicit MLP": "implicit"}


def _close(reported: float, actual: float) -> bool:
    """Report values are rounded for reading (3 decimals, or 4 significant
    digits for byte counts), so allow 5e-4 absolute below 1 and 0.5 percent
    relative above it."""
    if abs(actual) <= 1.0:
        return abs(reported - actual) <= 5e-4
    return abs(reported - actual) <= 5e-3 * abs(actual)


def _close_std(reported: float, actual: float) -> bool:
    return abs(reported - actual) <= 5e-4 or _close(reported, max(actual, 1e-9))


# Latency and tracemalloc are wall-clock measurements, so they differ between
# runs. Every other metric is seeded and must match the result file.
RUN_DEPENDENT = {"peak_mem_bytes", "update_ms", "query_ms"}


def _matches(value: float, key: str, mean, std) -> bool:
    if key not in RUN_DEPENDENT:
        return True
    if mean is None or std is None:
        return True
    return abs(value - mean) <= 0.25 * abs(mean) + 1e-9


def _table_rows(text: str):
    """Yield (row_label, [cell, ...]) for every markdown table row."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        yield cells[0], cells[1:]


def _load_metrics():
    out = {}
    with open(METRICS) as f:
        for row in csv.DictReader(f):
            out[(row["memory"], row["metric"])] = (
                float(row["mean"]) if row["mean"] else None,
                float(row["std"]) if row["std"] else None,
            )
    return out


def test_report_tables_match_result_files():
    assert REPORT.exists(), "REPORT.md missing"
    assert METRICS.exists(), "results/metrics.csv missing"
    metrics = _load_metrics()
    text = REPORT.read_text()

    checked = 0
    for label, cells in _table_rows(text):
        key = ROW_TO_KEY.get(label)
        if key is None or len(cells) != len(COL_TO_MEMORY):
            continue
        for cell, (col, memory) in zip(cells, COL_TO_MEMORY.items()):
            mean, std = metrics[(memory, key)]
            if cell == "-":
                assert mean is None, f"{label}/{col} shown as '-' but has a value"
                checked += 1
                continue
            m = re.fullmatch(r"(-?[\d.]+)\s*\+/-\s*(-?[\d.]+)", cell)
            assert m, f"unparsable cell {cell!r} in row {label!r}"
            rep_mean, rep_std = float(m.group(1)), float(m.group(2))
            checked += 1
            if key in RUN_DEPENDENT:
                # Only require that the reported latency or peak is plausible
                # for this machine, not identical to the last run.
                assert rep_mean > 0, f"{label}/{col}: {rep_mean} must be positive"
                assert _matches(rep_mean, key, mean, std), (
                    f"{label}/{col}: {rep_mean} far from {mean}"
                )
                continue
            assert _close(rep_mean, mean), f"{label}/{col}: {rep_mean} vs {mean}"
            assert _close_std(rep_std, std), f"{label}/{col} std: {rep_std} vs {std}"
    expected = 3 * (len(ROW_TO_KEY) - 3)  # minus the run-dependent rows
    assert checked >= expected, (
        f"only {checked} report cells checked; expected at least {expected}"
    )


def test_report_failure_case_matches_diagnostics():
    assert DIAGNOSTICS.exists(), "results/diagnostics.csv missing"
    with open(DIAGNOSTICS) as f:
        diag = {
            (int(r["n_objects"]), r["memory"]): float(r["current_success@2_mean"])
            for r in csv.DictReader(f)
        }
    text = REPORT.read_text()
    pairs = []
    for label, cells in _table_rows(text):
        if not label.isdigit() or len(cells) != 3:
            continue
        for i, memory in enumerate(("graph", "dense", "implicit")):
            pairs.append(((int(label), memory), float(cells[i])))
    assert len(pairs) >= 15, f"expected the 5x3 diagnostic table, got {len(pairs)} rows"
    for key, reported in pairs:
        assert _close(reported, diag[key]), f"{key}: report {reported} vs {diag[key]}"


def test_report_has_required_sections_and_figure():
    text = REPORT.read_text()
    assert "results/figure.png" in text, "report must reference the figure"
    assert (ROOT / "results" / "figure.png").exists()
    lowered = text.lower()
    assert "failure case" in lowered, "report must name a failure case"
    assert "limitations" in lowered, "report must have a limitations section"
    for memory in ("ObjectGraph", "DenseGrid", "Implicit"):
        assert memory in text
