"""Synthetic 3D grid world for the spatial-memory benchmark.

The world is deliberately simple so that the memory representation is the
only variable: a 16 x 16 x 8 grid holding 30 objects, an agent that moves to
a fresh random position every step, and a stochastic change process.

All randomness flows from explicitly passed seeds. Nothing else in the
benchmark touches global random state.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np

Obs = namedtuple("Obs", ["oid", "pos", "t", "conf"])
"""One observation: object id, estimated position (float array of shape (3,)),
world step, confidence (fixed at the detection probability)."""

GRID: Tuple[int, int, int] = (16, 16, 8)
GRID_LO = np.array([0.0, 0.0, 0.0])
GRID_HI = np.array([16.0, 16.0, 8.0])


class World:
    """A changing 3D grid world.

    Change process, applied once per step: with probability ``move_p`` one
    random object shifts by a N(0, move_sigma) offset per axis; with
    probability ``appear_p`` a new object with a fresh id appears; with
    probability ``remove_p`` a random existing object is removed.
    """

    def __init__(
        self,
        seed: int,
        n_objects: int = 30,
        grid: Tuple[int, int, int] = GRID,
        n_classes: int = 6,
        move_p: float = 0.10,
        appear_p: float = 0.05,
        remove_p: float = 0.03,
        detect_p: float = 0.8,
        noise_sigma: float = 0.2,
        view_radius: float = 6.0,
        occlusion_dist: float = 0.5,
        move_sigma: float = 0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.grid = tuple(int(g) for g in grid)
        self.n_classes = int(n_classes)
        self.move_p = float(move_p)
        self.appear_p = float(appear_p)
        self.remove_p = float(remove_p)
        self.detect_p = float(detect_p)
        self.noise_sigma = float(noise_sigma)
        self.view_radius = float(view_radius)
        self.occlusion_dist = float(occlusion_dist)
        self.move_sigma = float(move_sigma)

        self.objects: Dict[int, Tuple[np.ndarray, int]] = {}
        for oid in range(n_objects):
            pos = self.rng.uniform(GRID_LO, GRID_HI)
            cls = int(self.rng.integers(self.n_classes))
            self.objects[oid] = (pos, cls)
        self.next_id = int(n_objects)
        self.t = 0

    # -- changes -----------------------------------------------------------

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, GRID_LO, GRID_HI - 1e-9)

    def step(self) -> List[Tuple[str, int]]:
        """Advance one step and apply the change process.

        Returns the list of events applied at this step, e.g.
        ``[("move", 7), ("appear", 30)]``.
        """
        events: List[Tuple[str, int]] = []
        ids = sorted(self.objects)
        if ids and self.rng.random() < self.move_p:
            oid = ids[int(self.rng.integers(len(ids)))]
            pos, cls = self.objects[oid]
            self.objects[oid] = (
                self._clip(pos + self.rng.normal(0.0, self.move_sigma, 3)),
                cls,
            )
            events.append(("move", oid))
        if self.rng.random() < self.appear_p:
            pos = self.rng.uniform(GRID_LO, GRID_HI)
            cls = int(self.rng.integers(self.n_classes))
            self.objects[self.next_id] = (pos, cls)
            events.append(("appear", self.next_id))
            self.next_id += 1
        if ids and self.rng.random() < self.remove_p:
            oid = ids[int(self.rng.integers(len(ids)))]
            del self.objects[oid]
            events.append(("remove", oid))
        self.t += 1
        return events

    def snapshot(self) -> Dict[int, Tuple[np.ndarray, int]]:
        """Ground truth at the current step: id -> (position copy, class)."""
        return {oid: (pos.copy(), cls) for oid, (pos, cls) in self.objects.items()}

    # -- sensing -----------------------------------------------------------

    def observe(self, agent_pos) -> List[Obs]:
        """Observations visible from ``agent_pos`` at the current step.

        An object is a candidate when it lies within ``view_radius``. A
        candidate is dropped if another candidate closer to the agent blocks
        the line of sight within ``occlusion_dist``, then again with
        probability ``1 - detect_p``, and its reported position carries
        per-axis Gaussian noise of ``noise_sigma``.
        """
        agent = np.asarray(agent_pos, dtype=float)
        cand = []
        for oid, (pos, _cls) in self.objects.items():
            d = float(np.linalg.norm(pos - agent))
            if d <= self.view_radius:
                cand.append((d, oid, pos))
        cand.sort(key=lambda c: c[0])

        out: List[Obs] = []
        for i, (_d, oid, pos) in enumerate(cand):
            blockers = [c[2] for c in cand[:i]]
            if any(
                _occludes(agent, pos, b, self.occlusion_dist) for b in blockers
            ):
                continue
            if self.rng.random() > self.detect_p:
                continue
            est = self._clip(pos + self.rng.normal(0.0, self.noise_sigma, 3))
            out.append(Obs(oid=oid, pos=est, t=self.t, conf=self.detect_p))
        return out


def _occludes(agent: np.ndarray, target: np.ndarray, blocker: np.ndarray,
              occlusion_dist: float) -> bool:
    """True if ``blocker`` sits within ``occlusion_dist`` of the sight line
    agent -> target, in front of the target."""
    ab = target - agent
    denom = float(ab @ ab)
    if denom <= 0.0:
        return False
    s = float((blocker - agent) @ ab) / denom
    if s <= 0.0 or s >= 1.0:
        return False
    closest = agent + s * ab
    return float(np.linalg.norm(blocker - closest)) < occlusion_dist


def run_episode(seed: int, steps: int = 60, n_objects: int = 30):
    """Simulate one full episode.

    Returns ``(gt, obs_steps, agent_steps, world)`` where ``gt[t - 1]``,
    ``obs_steps[t - 1]`` and ``agent_steps[t - 1]`` correspond to world time
    ``t`` in ``1..steps``. The agent's random walk uses its own rng stream
    (seed + 100000) so world changes and agent motion can be replayed
    independently.
    """
    world = World(seed, n_objects=n_objects)
    arng = np.random.default_rng(100_000 + seed)
    gt: List[Dict[int, Tuple[np.ndarray, int]]] = []
    obs_steps: List[List[Obs]] = []
    agent_steps: List[np.ndarray] = []
    for _ in range(steps):
        world.step()
        agent = arng.uniform(GRID_LO, GRID_HI)
        obs_steps.append(world.observe(agent))
        gt.append(world.snapshot())
        agent_steps.append(agent.copy())
    return gt, obs_steps, agent_steps, world
