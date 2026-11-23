"""Tests for the extension experiments (capacity, eviction, relational)."""

import numpy as np
import pytest

from src.extensions import (
    CAPACITY_CONFIGS,
    EvictingDenseGridMemory,
    answer_left_of,
    answer_nearest_of_class,
    answer_near,
    answer_right_of,
    build_relational_schedule,
    evaluate_relational,
)
from src.memories import ImplicitMemory, ObjectGraphMemory
from src.metrics import build_schedule, evaluate_memory
from src.world import Obs, run_episode


def _obs(oid, pos, t):
    return Obs(oid=oid, pos=np.asarray(pos, float), t=t, conf=0.8)


# -- experiment A: capacity configuration ----------------------------------


def test_implicit_capacity_shapes_and_size_formula():
    for hidden, depth in CAPACITY_CONFIGS:
        m = ImplicitMemory(seed=3, hidden=hidden, depth=depth)
        assert m.W1.shape == (5, hidden)
        assert m.b1.shape == (hidden,)
        assert m.W2.shape == (hidden, 1)
        assert m.b2.shape == (1,)
        if depth == 2:
            assert m.W1b.shape == (hidden, hidden)
            assert m.b1b.shape == (hidden,)
        else:
            assert not hasattr(m, "W1b")
        m.update([_obs(0, [1.0, 1.0, 1.0], 1)], 1)
        m.update([_obs(0, [1.2, 1.0, 1.0], 2)], 2)
        assert all(np.all(np.isfinite(w)) for w in (m.W1, m.b1, m.W2, m.b2))
        weights = m.W1.size + m.b1.size + m.W2.size + m.b2.size
        if depth == 2:
            weights += m.W1b.size + m.b1b.size
        assert m.size_bytes() == 8 * weights + 48 * len(m.buffer) + 24 * len(
            m.last_seen
        )
        # The default configuration must keep the original 225 weights.
        if (hidden, depth) == (32, 1):
            assert weights == 225


def test_implicit_capacity_rejects_unsupported_values():
    with pytest.raises(ValueError):
        ImplicitMemory(hidden=16)
    with pytest.raises(ValueError):
        ImplicitMemory(depth=3)


def test_implicit_two_layer_training_changes_weights():
    m = ImplicitMemory(seed=4, hidden=32, depth=2, train_steps=5)
    before = m.W1b.copy()
    for t in range(1, 4):
        m.update([_obs(0, [2.0 + 0.1 * t, 3.0, 1.0], t),
                  _obs(1, [8.0, 8.0, 4.0], t)], t)
    assert not np.allclose(before, m.W1b)


def test_implicit_two_layer_backprop_matches_finite_differences():
    m = ImplicitMemory(seed=0, hidden=32, depth=2, train_steps=0)
    rng = np.random.default_rng(13)
    X = rng.uniform(0.0, 1.0, (8, 5))
    y = (rng.uniform(0.0, 1.0, (8, 1)) < 0.5).astype(float)

    def loss(W1, W1b, W2):
        H1 = np.tanh(X @ W1 + m.b1)
        H2 = np.tanh(H1 @ W1b + m.b1b)
        p = 1.0 / (1.0 + np.exp(-(H2 @ W2 + m.b2)))
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps)
                             + (1 - y) * np.log(1 - p + eps)))

    H1, H2, p = m._forward_with_hidden(X)
    gz = (p - y) / X.shape[0]
    gW2 = H2.T @ gz
    gH2 = gz @ m.W2.T
    gH2t = gH2 * (1.0 - H2 * H2)
    gW1b = H1.T @ gH2t
    gH1 = gH2t @ m.W1b.T
    gH1t = gH1 * (1.0 - H1 * H1)
    gW1 = X.T @ gH1t

    eps = 1e-6
    checks = (
        (m.W1, gW1, lambda W: loss(W, m.W1b, m.W2), (5, 32)),
        (m.W1b, gW1b, lambda W: loss(m.W1, W, m.W2), (32, 32)),
        (m.W2, gW2, lambda W: loss(m.W1, m.W1b, W), (32, 1)),
    )
    for value, grad, evaluate, shape in checks:
        for _ in range(4):
            i, j = rng.integers(0, shape[0]), rng.integers(0, shape[1])
            plus = value.copy()
            minus = value.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            num = (evaluate(plus) - evaluate(minus)) / (2 * eps)
            assert abs(num - grad[i, j]) < 1e-6


# -- experiment C: eviction -------------------------------------------------


def test_eviction_drops_only_voxels_older_than_max_age():
    m = EvictingDenseGridMemory(max_age=10)
    m.update([_obs(0, [1.2, 1.2, 1.2], 1)], 1)
    assert m.query_current(0, 10) is not None  # age 9
    m.update([], 11)  # age 10: still kept
    assert m.query_current(0, 11) is not None
    m.update([], 12)  # age 11: dropped
    assert m.query_current(0, 12) is None
    assert m.evicted_voxels == 1
    # The arrays keep their shape, so persistent size is unchanged.
    assert m.size_bytes() == 16 * 16 * 8 * (4 + 8)


def test_eviction_prevents_collision_with_cleared_voxel():
    m = EvictingDenseGridMemory(max_age=1)
    m.update([_obs(0, [1.2, 1.2, 1.2], 1)], 1)
    m.update([_obs(1, [1.4, 1.4, 1.4], 2)], 2)  # same voxel: collision
    assert m.collisions == 1
    m.update([], 4)  # age 2 > 1: voxel cleared
    m.update([_obs(2, [1.1, 1.1, 1.1], 5)], 5)
    assert m.collisions == 1  # no collision against the cleared voxel


# -- experiment B: relational semantics ------------------------------------


def test_pairwise_relational_semantics():
    a = np.array([1.0, 4.0, 2.0])
    b = np.array([5.0, 1.0, 2.0])
    assert answer_left_of(a, b) is True
    assert answer_left_of(b, a) is False
    assert answer_right_of(b, a) is True
    assert answer_right_of(a, b) is False
    assert answer_near(np.array([0.0, 0.0, 0.0]),
                       np.array([3.0, 0.0, 0.0])) is True
    assert answer_near(np.array([0.0, 0.0, 0.0]),
                       np.array([3.01, 0.0, 0.0])) is False
    assert answer_left_of(None, b) is None
    assert answer_right_of(a, None) is None
    assert answer_near(None, b) is None


def test_nearest_of_class_filters_and_breaks_ties():
    positions = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([4.0, 0.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0]),
    }
    classes = {0: 1, 1: 0, 2: 0}
    assert answer_nearest_of_class(positions, classes, 0,
                                   [1.5, 0.0, 0.0]) == 2
    assert answer_nearest_of_class(positions, classes, 1,
                                   [1.5, 0.0, 0.0]) == 0
    assert answer_nearest_of_class(positions, classes, 9,
                                   [1.5, 0.0, 0.0]) is None
    # Equidistant class members resolve to the lower id.
    tie = {5: np.array([0.0, 0.0, 0.0]), 3: np.array([2.0, 0.0, 0.0])}
    assert answer_nearest_of_class(tie, {5: 0, 3: 0}, 0,
                                   [1.0, 0.0, 0.0]) == 3


def test_relational_schedule_and_evaluation_are_deterministic():
    gt, obs_steps, _agents, _world = run_episode(0, steps=30)
    sched = build_relational_schedule(gt, seed=0, n_per_type=5)
    for kind in ("left_of", "right_of", "near", "nearest_of_class"):
        assert len(sched[kind]) == 5
    for q in sched["left_of"]:
        left = gt[q["t"] - 1][q["a"]][0][0]
        right = gt[q["t"] - 1][q["b"]][0][0]
        assert q["truth"] == (left < right)

    res = evaluate_relational(ObjectGraphMemory(), obs_steps, sched)
    res2 = evaluate_relational(ObjectGraphMemory(), obs_steps, sched)
    for kind in ("left_of", "right_of", "near", "nearest_of_class"):
        assert res[f"{kind}_n"] == 5
        assert 0.0 <= res[f"{kind}_accuracy"] <= 1.0
        assert res[f"{kind}_accuracy"] == res2[f"{kind}_accuracy"]


def test_latency_checkpoint_is_optional_and_reported():
    gt, obs_steps, _agents, _world = run_episode(2, steps=20)
    sched = build_schedule(gt, obs_steps, seed=2, n_per_type=3)
    plain = evaluate_memory(ObjectGraphMemory(), obs_steps, sched)
    assert "query_ms@20" not in plain
    checked = evaluate_memory(
        ObjectGraphMemory(),
        obs_steps,
        sched,
        growth_steps=(20,),
        latency_checkpoints=(20,),
        latency_window=10,
    )
    assert checked["growth@20"] is not None
    assert checked["query_ms@20"] is not None and checked["query_ms@20"] >= 0.0
