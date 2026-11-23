"""Query schedule and metric tests."""

import numpy as np

from src.metrics import build_schedule, evaluate_memory
from src.world import run_episode


def test_schedule_shapes_and_ranges():
    gt, obs_steps, _agents, _world = run_episode(0, steps=60)
    sched = build_schedule(gt, obs_steps, seed=0, n_per_type=25)
    assert len(sched["current"]) == 25
    assert len(sched["past"]) == 25
    assert len(sched["region"]) == 25
    assert len(sched["stale"]) == 25
    for t, oid, true in sched["current"]:
        assert 1 <= t <= 60 and oid in gt[t - 1]
        assert np.allclose(true, gt[t - 1][oid][0])
    for t, oid, _true in sched["past"]:
        assert 11 <= t <= 60 and oid in gt[t - 11]
    for t, _bounds, changed in sched["region"]:
        assert 11 <= t <= 60
        assert isinstance(changed, set)
    for t, gt_stale in sched["stale"]:
        assert 15 <= t <= 60
        assert gt_stale  # sampler prefers non-empty ground truth
    # Deterministic in the seed.
    sched2 = build_schedule(gt, obs_steps, seed=0, n_per_type=25)
    assert [q[:2] for q in sched["current"]] == [q[:2] for q in sched2["current"]]


def test_evaluate_memory_produces_full_metric_set():
    gt, obs_steps, _agents, _world = run_episode(2, steps=60)
    sched = build_schedule(gt, obs_steps, seed=2, n_per_type=10)
    from src.memories import ObjectGraphMemory

    res = evaluate_memory(ObjectGraphMemory(), obs_steps, sched)
    expected = {
        "current_success@2", "past_success@2", "region_precision",
        "region_recall", "stale_precision", "stale_recall", "stale_answer_rate",
        "size_bytes", "peak_mem_bytes", "update_ms", "query_ms",
        "growth@10", "growth@30", "growth@60",
    }
    assert expected <= set(res)
    for key in ("current_success@2", "past_success@2"):
        assert 0.0 <= res[key] <= 1.0
    # The graph memory answers from the latest observation, so current queries
    # fail only for objects it has not seen for a while. With 10 samples the
    # fraction is noisy; the benchmark uses 25 per seed.
    assert res["current_success@2"] >= 0.7
    assert res["size_bytes"] > 0 and res["update_ms"] >= 0


def test_short_episode_skips_unsupported_query_types():
    gt, obs_steps, _agents, _world = run_episode(5, steps=8)
    sched = build_schedule(gt, obs_steps, seed=5, n_per_type=4)
    assert len(sched["current"]) == 4
    assert sched["past"] == []
    assert sched["region"] == []
    assert sched["stale"] == []
    from src.memories import DenseGridMemory

    res = evaluate_memory(DenseGridMemory(), obs_steps, sched,
                          growth_steps=(10, 30, 60))
    assert res["past_success@2"] is None
    assert res["growth@10"] is None  # episode never reaches step 10
