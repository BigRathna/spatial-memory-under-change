"""World simulation tests."""

import numpy as np

from src.world import GRID_HI, GRID_LO, Obs, World, _occludes, run_episode


def test_episode_deterministic():
    a = run_episode(0, steps=30)
    b = run_episode(0, steps=30)
    assert len(a[0]) == len(b[0]) == 30
    for ta, tb in zip(a[0], b[0]):
        assert sorted(ta) == sorted(tb)
        for oid in ta:
            assert np.allclose(ta[oid][0], tb[oid][0])
            assert ta[oid][1] == tb[oid][1]
    for oa, ob in zip(a[1], b[1]):
        assert len(oa) == len(ob)
        for x, y in zip(oa, ob):
            assert x.oid == y.oid and x.t == y.t
            assert np.allclose(x.pos, y.pos)


def test_change_events_occur_over_long_run():
    world = World(3)
    seen = {"move": 0, "appear": 0, "remove": 0}
    n_objects = []
    for _ in range(600):
        for ev in world.step():
            seen[ev[0]] += 1
        n_objects.append(len(world.objects))
    # With the spec rates (0.10 / 0.05 / 0.03) over 600 steps each event
    # type must have fired at least once for this fixed seed.
    assert seen["move"] > 0
    assert seen["appear"] > 0
    assert seen["remove"] > 0
    # Object count stays bounded and ids only grow.
    assert 0 < min(n_objects)
    assert max(n_objects) <= 30 + seen["appear"]


def test_observation_geometry():
    gt, obs_steps, agent_steps, _world = run_episode(1, steps=60)
    for t, (obs_list, agent) in enumerate(zip(obs_steps, agent_steps), start=1):
        present = set(gt[t - 1])
        for o in obs_list:
            assert o.t == t
            assert o.oid in present
            true_pos = gt[t - 1][o.oid][0]
            d = np.linalg.norm(o.pos - agent)
            # Observation noise is N(0, 0.2) per axis; 6 sigma is a safe bound.
            assert d <= 6.0 + 6 * 0.2
            assert np.all(o.pos >= GRID_LO) and np.all(o.pos < GRID_HI)
            assert o.conf == 0.8


def test_occlusion_helper():
    agent = np.zeros(3)
    target = np.array([4.0, 0.0, 0.0])
    blocker_mid = np.array([2.0, 0.2, 0.0])  # on the sight line
    assert _occludes(agent, target, blocker_mid, 0.5)
    blocker_side = np.array([2.0, 3.0, 0.0])  # far off the sight line
    assert not _occludes(agent, target, blocker_side, 0.5)
    behind = np.array([5.0, 0.0, 0.0])  # past the target
    assert not _occludes(agent, target, behind, 0.5)


def test_occlusion_drops_farther_object():
    # Two objects on the same ray from the agent, closer than the view
    # radius, with detection forced on: only the near one is observed.
    world = World(7, n_objects=0, detect_p=1.0, noise_sigma=0.0,
                  view_radius=6.0, occlusion_dist=0.5)
    world.objects = {
        0: (np.array([2.0, 8.0, 4.0]), 1),
        1: (np.array([4.0, 8.0, 4.0]), 2),
    }
    world.t = 1
    agent = np.array([0.0, 8.0, 4.0])
    obs = world.observe(agent)
    assert [o.oid for o in obs] == [0]
