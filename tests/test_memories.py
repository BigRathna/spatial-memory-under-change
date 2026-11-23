"""Memory representation tests."""

import numpy as np

from src.memories import DenseGridMemory, ImplicitMemory, ObjectGraphMemory
from src.world import Obs


def _obs(oid, pos, t):
    return Obs(oid=oid, pos=np.asarray(pos, float), t=t, conf=0.8)


def _feed(mem, calls):
    for t, obs_list in calls:
        mem.update(obs_list, t)


# -- object graph ----------------------------------------------------------


def test_graph_query_current_and_at():
    m = ObjectGraphMemory()
    _feed(m, [
        (1, [_obs(0, [1.0, 1.0, 1.0], 1)]),
        (5, [_obs(0, [3.0, 2.0, 1.0], 5)]),
    ])
    pos, last, conf = m.query_current(0, 6)
    assert np.allclose(pos, [3.0, 2.0, 1.0]) and last == 5 and conf == 0.8
    # Time-filtered: at t=2 only the first record is visible.
    pos, last, _ = m.query_current(0, 2)
    assert np.allclose(pos, [1.0, 1.0, 1.0]) and last == 1
    assert m.query_at(0, 2) is not None and np.allclose(m.query_at(0, 2), [1.0, 1.0, 1.0])
    assert m.query_current(0, 0) is None
    assert m.query_current(99, 6) is None


def test_graph_region_and_stale():
    m = ObjectGraphMemory()
    _feed(m, [
        (1, [_obs(0, [1.0, 1.0, 1.0], 1), _obs(1, [9.0, 9.0, 4.0], 1)]),
        (2, [_obs(1, [9.5, 9.5, 4.5], 2)]),
    ])
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([4.0, 4.0, 4.0])
    assert m.query_region((lo, hi), 5) == [0]
    assert m.query_region((lo, np.array([16.0, 16.0, 8.0])), 5) == [0, 1]
    assert m.stale_ids(t=20, max_age=15) == [0, 1]
    assert m.stale_ids(t=16, max_age=15) == [0]  # id 1 last seen at step 2


def test_graph_grows_with_updates():
    m = ObjectGraphMemory()
    s0 = m.size_bytes()
    m.update([_obs(0, [1.0, 1.0, 1.0], 1)], 1)
    s1 = m.size_bytes()
    m.update([_obs(0, [2.0, 1.0, 1.0], 2)], 2)
    s2 = m.size_bytes()
    assert s0 < s1 < s2


# -- dense grid ------------------------------------------------------------


def test_dense_query_and_overwrite_collision():
    m = DenseGridMemory()
    m.update([_obs(0, [1.4, 1.4, 1.4], 1)], 1)
    pos, last, _ = m.query_current(0, 2)
    assert np.allclose(pos, [1.0, 1.0, 1.0]) and last == 1
    # Same voxel, different object: the later write wins, id 0 vanishes.
    m.update([_obs(7, [1.2, 1.2, 1.2], 3)], 3)
    assert m.query_current(0, 4) is None
    assert m.query_current(7, 4) is not None
    assert m.collisions == 1


def test_dense_region_and_stale():
    m = DenseGridMemory()
    _feed(m, [
        (1, [_obs(0, [1.5, 1.5, 1.5], 1), _obs(5, [10.0, 10.0, 5.0], 1)]),
        (4, [_obs(0, [1.6, 1.6, 1.6], 4)]),
    ])
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([3.0, 3.0, 3.0])
    assert m.query_region((lo, hi), 10) == [0]
    assert m.query_region((np.zeros(3), np.array([16.0, 16.0, 8.0])), 10) == [0, 5]
    # id 5 last seen at step 1: stale at t=16 with max_age 15.
    assert m.stale_ids(t=16, max_age=15) == [5]
    # at t=19 the cutoff is 4, so id 0 (last seen at step 4) is stale too.
    assert m.stale_ids(t=19, max_age=15) == [0, 5]
    # Size is fixed by the grid shape.
    assert m.size_bytes() == 16 * 16 * 8 * (4 + 8)


# -- implicit memory -------------------------------------------------------


def test_implicit_backprop_matches_finite_differences():
    m = ImplicitMemory(seed=0, train_steps=0)
    rng = np.random.default_rng(11)
    X = rng.uniform(0.0, 1.0, (8, 5))
    y = (rng.uniform(0.0, 1.0, (8, 1)) < 0.5).astype(float)

    def loss(W1, W2):
        H = np.tanh(X @ W1 + m.b1)
        p = 1.0 / (1.0 + np.exp(-(H @ W2 + m.b2)))
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    H, p = m._forward(X)
    gz = (p - y) / X.shape[0]
    gW2 = H.T @ gz
    gH = gz @ m.W2.T
    gHt = gH * (1.0 - H * H)
    gW1 = X.T @ gHt

    eps = 1e-6
    for _ in range(6):
        i, j = rng.integers(0, 5), rng.integers(0, 32)
        Wp = m.W1.copy(); Wp[i, j] += eps
        Wm = m.W1.copy(); Wm[i, j] -= eps
        num = (loss(Wp, m.W2) - loss(Wm, m.W2)) / (2 * eps)
        assert abs(num - gW1[i, j]) < 1e-6
    for _ in range(6):
        i, j = rng.integers(0, 32), rng.integers(0, 1)
        Wp = m.W2.copy(); Wp[i, j] += eps
        Wm = m.W2.copy(); Wm[i, j] -= eps
        num = (loss(m.W1, Wp) - loss(m.W1, Wm)) / (2 * eps)
        assert abs(num - gW2[i, j]) < 1e-6


def test_implicit_learns_static_object():
    m = ImplicitMemory(seed=1)
    true = np.array([3.2, 4.1, 1.5])
    for t in range(1, 31):
        m.update([_obs(0, true + np.full(3, 0.02 * t), t)], t)
    ans = m.query_current(0, 31)
    assert ans is not None, "MLP should clear the 0.5 occupancy threshold"
    pos, last_seen, p = ans
    assert last_seen == 30
    assert np.linalg.norm(pos - true) <= 2.0
    assert p >= 0.5
    # The scan must not be uniformly high: the bump is localized.
    allp = m._occupancy(m._cells, 31, 0)
    assert (allp > 0.5).mean() < 0.2


def test_implicit_rejects_unknown_id_and_size_formula():
    m = ImplicitMemory(seed=2, train_steps=0)
    assert m.query_current(123, 5) is None
    m.update([_obs(4, [1.0, 1.0, 1.0], 1)], 1)
    n_weights = m.W1.size + m.b1.size + m.W2.size + m.b2.size
    assert m.size_bytes() == 8 * n_weights + 48 * len(m.buffer) + 24 * len(m.last_seen)
    assert m.stale_ids(t=20, max_age=15) == [4]
