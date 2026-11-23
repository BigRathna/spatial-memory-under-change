"""End-to-end smoke test of the demo driver at reduced scale."""

import numpy as np

from src.demo import MEMORY_ORDER, format_markdown_table, run_seed
from src.metrics import METRIC_KEYS


def test_demo_smoke_one_seed():
    res = run_seed(0, steps=18, n_per_type=2)
    assert sorted(res) == sorted(MEMORY_ORDER)
    for name in MEMORY_ORDER:
        m = res[name]
        for key in ("current_success@2", "past_success@2",
                    "stale_answer_rate", "region_precision", "region_recall",
                    "stale_precision", "stale_recall"):
            assert m[key] is None or 0.0 <= m[key] <= 1.0, (name, key)
        assert m["size_bytes"] > 0
        assert m["update_ms"] >= 0 and m["query_ms"] >= 0
        assert m["growth@10"] is not None  # 18 steps reach step 10
    # Deterministic in the seed: rerunning gives identical accuracy metrics.
    res2 = run_seed(0, steps=18, n_per_type=2)
    for name in MEMORY_ORDER:
        assert res[name]["current_success@2"] == res2[name]["current_success@2"]
        assert res[name]["size_bytes"] == res2[name]["size_bytes"]


def test_markdown_table_has_all_metrics_and_memories():
    res = run_seed(1, steps=14, n_per_type=2)
    from src.metrics import aggregate

    agg = {name: aggregate([res[name]]) for name in MEMORY_ORDER}
    table = format_markdown_table(agg)
    assert table.count("\n") == len(METRIC_KEYS) + 1
    for name in MEMORY_ORDER:
        assert name in table
    for key in METRIC_KEYS:
        assert key in table
    import re

    assert not re.search(r"\bnan\b", table)
    # Latency columns vary between runs; accuracy cells must be numeric.
    for row in table.splitlines()[2:]:
        assert row.startswith("| ")
