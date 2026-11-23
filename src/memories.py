"""Three spatial-memory representations behind one interface.

``size_bytes`` estimates *persistent* size: the bytes a serialization of the
state would need on disk, not live Python object overhead. The estimate used
by each memory is documented in its class docstring and applied consistently
across the three implementations.

Query semantics shared by all memories:

- ``query_current(oid, t)`` -> ``(position, last_seen, confidence)`` or
  ``None``: the memory's best guess for the object's position at time ``t``,
  using only information with step <= t.
- ``query_at(oid, t)`` -> position or ``None``: position at or before ``t``.
- ``query_region(bounds, t)`` -> ids whose best-known position at or before
  ``t`` lies inside the axis-aligned box ``bounds = (lo, hi)`` (inclusive on
  both ends, one 3-vector per corner).
- ``stale_ids(t, max_age)`` -> tracked ids whose last observation is at
  least ``max_age`` steps old (``t - last_seen >= max_age``).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .world import GRID, Obs


class Memory:
    """Common interface for the three spatial-memory representations."""

    def update(self, observations: Iterable[Obs], t: int) -> None: ...

    def query_current(self, object_id: int, t: int): ...

    def query_at(self, object_id: int, t: int): ...

    def query_region(self, bounds, t: int) -> List[int]: ...

    def stale_ids(self, t: int, max_age: int) -> List[int]: ...

    def size_bytes(self) -> int: ...


# ---------------------------------------------------------------------------
# 1. Object graph
# ---------------------------------------------------------------------------


class ObjectGraphMemory(Memory):
    """Object id -> list of (position, t, confidence).

    Queries scan the record list (newest first). Strong on identity and
    semantic queries; the record store grows without bound.

    Persistent-size estimate: 48 bytes per record (3 float64 coordinates, one
    int64 step, one float64 confidence, one index slot) plus 16 bytes per
    tracked id and a 64-byte header.
    """

    def __init__(self) -> None:
        self.records: Dict[int, List[Tuple[np.ndarray, int, float]]] = {}

    def update(self, observations: Iterable[Obs], t: int) -> None:
        t = int(t)
        for o in observations:
            self.records.setdefault(o.oid, []).append(
                (np.asarray(o.pos, dtype=float).copy(), t, float(o.conf))
            )

    def _latest(self, oid: int, t: int):
        recs = self.records.get(oid)
        if not recs:
            return None
        for pos, tr, conf in reversed(recs):
            if tr <= t:
                return pos, tr, conf
        return None

    def query_current(self, object_id: int, t: int):
        r = self._latest(object_id, t)
        if r is None:
            return None
        pos, tr, conf = r
        return (pos.copy(), tr, conf)

    def query_at(self, object_id: int, t: int) -> Optional[np.ndarray]:
        r = self._latest(object_id, t)
        return None if r is None else r[0].copy()

    def query_region(self, bounds, t: int) -> List[int]:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        out = []
        for oid in self.records:
            r = self._latest(oid, t)
            if r is None:
                continue
            pos = r[0]
            if bool(np.all(pos >= lo)) and bool(np.all(pos <= hi)):
                out.append(oid)
        return out

    def stale_ids(self, t: int, max_age: int) -> List[int]:
        cutoff = int(t) - int(max_age)
        return [oid for oid, recs in self.records.items() if recs[-1][1] <= cutoff]

    def size_bytes(self) -> int:
        n_records = sum(len(v) for v in self.records.values())
        return 64 + 48 * n_records + 16 * len(self.records)


# ---------------------------------------------------------------------------
# 2. Dense grid
# ---------------------------------------------------------------------------


class DenseGridMemory(Memory):
    """A dense (I, J, K) int32 id array plus a parallel int64 last-seen array.

    Each observation writes the object id and step into the voxel containing
    the observed position. When two objects share a voxel the later write
    wins and the earlier id disappears from the map; such overwrites are
    counted in ``collisions``. Voxels are never cleared when an object moves
    away, so an id can occupy several voxels at once (ghosts). The map stores
    no time series: it only keeps the latest write per voxel.

    Persistent size is exact: the two arrays, fixed by the grid shape.
    """

    def __init__(self, grid: Tuple[int, int, int] = GRID) -> None:
        self.grid = tuple(int(g) for g in grid)
        self.ids = np.full(self.grid, -1, dtype=np.int32)
        self.lastseen = np.full(self.grid, -1, dtype=np.int64)
        self.collisions = 0

    def _voxel(self, pos) -> Tuple[int, int, int]:
        v = np.floor(np.asarray(pos, dtype=float)).astype(int)
        v = np.clip(v, 0, np.asarray(self.grid) - 1)
        return (int(v[0]), int(v[1]), int(v[2]))

    def update(self, observations: Iterable[Obs], t: int) -> None:
        t = int(t)
        for o in observations:
            v = self._voxel(o.pos)
            prev = int(self.ids[v])
            if prev != -1 and prev != int(o.oid):
                self.collisions += 1
            self.ids[v] = int(o.oid)
            self.lastseen[v] = t

    def _cells(self, bounds):
        """Slices covering all voxels that intersect the box ``bounds``."""
        g = self.grid
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        sl = []
        for a in range(3):
            i0 = int(np.clip(np.floor(lo[a]), 0, g[a] - 1))
            i1 = int(np.clip(np.floor(hi[a]), -1, g[a] - 1))
            if i1 < i0:
                return None
            sl.append(slice(i0, i1 + 1))
        return tuple(sl)

    def query_current(self, object_id: int, t: int):
        t = int(t)
        valid = (
            (self.ids == int(object_id))
            & (self.lastseen >= 0)
            & (self.lastseen <= t)
        )
        if not valid.any():
            return None
        ls = np.where(valid, self.lastseen, -1)
        v = np.unravel_index(int(np.argmax(ls)), self.grid)
        return (np.array(v, dtype=float), int(self.lastseen[v]), 1.0)

    def query_at(self, object_id: int, t: int) -> Optional[np.ndarray]:
        r = self.query_current(object_id, t)
        return None if r is None else r[0]

    def query_region(self, bounds, t: int) -> List[int]:
        t = int(t)
        sl = self._cells(bounds)
        if sl is None:
            return []
        m = self.ids[sl]
        ls = self.lastseen[sl]
        mask = (m != -1) & (ls >= 0) & (ls <= t)
        return sorted({int(x) for x in m[mask]})

    def stale_ids(self, t: int, max_age: int) -> List[int]:
        # Per id, the last observation is the newest of its occupied voxels.
        # Testing voxels individually would mark an id stale whenever any
        # ghost voxel is old, even if the object was seen a step ago.
        cutoff = int(t) - int(max_age)
        mask = (self.ids >= 0) & (self.lastseen >= 0)
        newest: Dict[int, int] = {}
        for oid, ts in zip(self.ids[mask].tolist(), self.lastseen[mask].tolist()):
            if ts > newest.get(int(oid), -1):
                newest[int(oid)] = int(ts)
        return sorted(oid for oid, ts in newest.items() if ts <= cutoff)

    def size_bytes(self) -> int:
        return int(self.ids.nbytes + self.lastseen.nbytes)


# ---------------------------------------------------------------------------
# 3. Implicit coordinate MLP
# ---------------------------------------------------------------------------


class ImplicitMemory(Memory):
    """A coordinate MLP over (x, y, z, t, object_id).

    Architecture: 5 -> ``hidden`` tanh -> [``hidden`` tanh ->] 1 sigmoid
    occupancy probability, where ``hidden`` is 32, 64 or 128 and the second
    hidden layer is present only when ``depth = 2``. The default
    (32, depth 1) is the configuration used by the original benchmark.
    Training is online with manual backprop and SGD (50 steps per update,
    learning rate 0.01, batch 32). Each batch is half positives drawn from
    the replay
    buffer and half negatives carrying the *same* id and step: half of the
    negatives are sampled within +-3 cells of a positive (hard negatives,
    which force a localized bump) and half uniformly over the grid (which
    suppresses distant false positives). Coordinates enter scaled as
    ``x / grid * 3`` so one cell is a ~0.2 change in the input, which the
    32-unit layer can resolve in 50 steps. Loss is binary cross-entropy
    summed over the batch, not averaged: with a mean, the per-update weight
    change at learning rate 0.01 is too small to learn anything within the
    50-step budget, and the model collapses to a corner-biased solution.

    This is a small implicit function, not a NeRF: no view direction, no ray
    marching, ~1.8 KB of weights, trained on CPU in seconds.

    Positives are drawn from the buffer with recency weighting
    (``exp(-age / TAU)``) rather than uniformly. Uniform replay trains the
    model to reproduce *historical* positions, so ``query_current`` returns
    the ghost of wherever the object used to be; the query contract asks for
    the present, and recency weighting is the minimum fix for that.

    Position queries scan the coarse grid (integer cells, step 1.0) for the
    given id and step and return the highest-probability cell if it clears
    ``threshold``.

    Persistent-size estimate: 8 bytes per weight (225 for the default
    configuration), 48 bytes per replay-buffer entry, 24 bytes per tracked id
    in the last-seen map.
    """

    IN_DIM = 5
    HIDDEN = 32
    HIDDEN_CHOICES = (32, 64, 128)
    DEPTH_CHOICES = (1, 2)
    LR = 0.01
    TRAIN_STEPS = 50
    BATCH = 32
    T_SCALE = 64.0
    ID_SCALE = 64.0
    COORD_SCALE = 3.0
    LOCAL_NEG_FRAC = 0.5
    LOCAL_NEG_RADIUS = 3.0
    REPLAY_TAU = 2.0

    def __init__(
        self,
        seed: int = 0,
        grid: Tuple[int, int, int] = GRID,
        lr: float = LR,
        train_steps: int = TRAIN_STEPS,
        batch: int = BATCH,
        buffer_size: int = 200,
        threshold: float = 0.5,
        hidden: int = HIDDEN,
        depth: int = 1,
    ) -> None:
        if int(hidden) not in self.HIDDEN_CHOICES:
            raise ValueError(
                f"hidden must be one of {self.HIDDEN_CHOICES}, got {hidden!r}"
            )
        if int(depth) not in self.DEPTH_CHOICES:
            raise ValueError(
                f"depth must be one of {self.DEPTH_CHOICES}, got {depth!r}"
            )
        self.rng = np.random.default_rng(seed)
        self.grid = np.asarray(grid, dtype=float)
        self.lr = float(lr)
        self.train_steps = int(train_steps)
        self.batch = int(batch)
        self.buffer_size = int(buffer_size)
        self.threshold = float(threshold)
        self.hidden = int(hidden)
        self.depth = int(depth)

        self.W1 = self.rng.normal(0.0, 1.0 / np.sqrt(self.IN_DIM),
                                  (self.IN_DIM, self.hidden))
        self.b1 = np.zeros(self.hidden)
        if self.depth == 2:
            self.W1b = self.rng.normal(0.0, 1.0 / np.sqrt(self.hidden),
                                       (self.hidden, self.hidden))
            self.b1b = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0.0, 1.0 / np.sqrt(self.hidden),
                                  (self.hidden, 1))
        self.b2 = np.zeros(1)

        self.buffer: deque = deque(maxlen=self.buffer_size)
        self.last_seen: Dict[int, int] = {}

        gx, gy, gz = (int(v) for v in grid)
        xs, ys, zs = np.arange(gx), np.arange(gy), np.arange(gz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self._cells = np.stack(
            [X.ravel(), Y.ravel(), Z.ravel()], axis=1
        ).astype(float)

    # -- model -------------------------------------------------------------

    def _encode(self, xyz: np.ndarray, t, oid) -> np.ndarray:
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        n = xyz.shape[0]
        ts = np.broadcast_to(np.asarray(t, dtype=float).reshape(-1), (n,))
        ids = np.broadcast_to(np.asarray(oid, dtype=float).reshape(-1), (n,))
        return np.column_stack(
            [
                xyz[:, 0] / self.grid[0] * self.COORD_SCALE,
                xyz[:, 1] / self.grid[1] * self.COORD_SCALE,
                xyz[:, 2] / self.grid[2] * self.COORD_SCALE,
                ts / self.T_SCALE,
                ids / self.ID_SCALE,
            ]
        )

    def _forward_with_hidden(self, X: np.ndarray):
        H1 = np.tanh(X @ self.W1 + self.b1)
        if self.depth == 2:
            H2 = np.tanh(H1 @ self.W1b + self.b1b)
        else:
            H2 = H1
        z = H2 @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z))
        return H1, H2, p

    def _forward(self, X: np.ndarray):
        _H1, H, p = self._forward_with_hidden(X)
        return H, p

    def _occupancy(self, xyz, t, oid) -> np.ndarray:
        X = self._encode(xyz, np.full(len(xyz), float(t)),
                         np.full(len(xyz), float(oid)))
        _H, p = self._forward(X)
        return p[:, 0]

    def _sample_negatives(self, pos: np.ndarray, n: int) -> np.ndarray:
        """``n`` negative positions: half hard (within LOCAL_NEG_RADIUS of a
        positive), half uniform over the grid."""
        n_local = int(round(n * self.LOCAL_NEG_FRAC))
        parts = []
        if n_local:
            # Pair each hard negative with one of the positives already drawn.
            k = min(n_local, pos.shape[0])
            anchors = pos[np.arange(n_local) % k]
            lo = np.clip(anchors - self.LOCAL_NEG_RADIUS, 0.0, self.grid)
            hi = np.clip(anchors + self.LOCAL_NEG_RADIUS, 0.0, self.grid)
            parts.append(self.rng.uniform(lo, hi))
        if n - n_local:
            parts.append(self.rng.uniform(0.0, self.grid, size=(n - n_local, 3)))
        return np.vstack(parts)

    # -- interface ---------------------------------------------------------

    def update(self, observations: Iterable[Obs], t: int) -> None:
        t = int(t)
        for o in observations:
            self.buffer.append(
                (np.asarray(o.pos, dtype=float).copy(), t, int(o.oid))
            )
            self.last_seen[int(o.oid)] = t
        self._train()

    def _train(self) -> None:
        n = len(self.buffer)
        if n < 2:
            return
        buf = list(self.buffer)
        ts_all = np.array([float(b[1]) for b in buf])
        weights = np.exp(-(ts_all.max() - ts_all) / self.REPLAY_TAU)
        weights = weights / weights.sum()
        half = self.batch // 2
        for _ in range(self.train_steps):
            idx = self.rng.choice(n, size=half, p=weights)
            pos = np.stack([buf[i][0] for i in idx])
            ts = ts_all[idx]
            ids = np.array([float(buf[i][2]) for i in idx])
            Xp = self._encode(pos, ts, ids)
            neg = self._sample_negatives(pos, half)
            Xn = self._encode(neg, ts, ids)
            X = np.vstack([Xp, Xn])
            y = np.vstack([np.ones((half, 1)), np.zeros((half, 1))])

            H1, H2, p = self._forward_with_hidden(X)
            # Binary cross-entropy, sigmoid output: dL/dz = (p - y), summed
            # over the batch (no 1/B factor, see the class docstring).
            gz = p - y
            gW2 = H2.T @ gz
            gb2 = gz.sum(axis=0)
            gH2 = gz @ self.W2.T
            if self.depth == 2:
                gH2t = gH2 * (1.0 - H2 * H2)
                gW1b = H1.T @ gH2t
                gb1b = gH2t.sum(axis=0)
                gH1 = gH2t @ self.W1b.T
                gH1t = gH1 * (1.0 - H1 * H1)
            else:
                gH1t = gH2 * (1.0 - H2 * H2)
            gW1 = X.T @ gH1t
            gb1 = gH1t.sum(axis=0)

            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
            if self.depth == 2:
                self.W1b -= self.lr * gW1b
                self.b1b -= self.lr * gb1b
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2

    def query_current(self, object_id: int, t: int):
        oid = int(object_id)
        if oid not in self.last_seen:
            return None
        p = self._occupancy(self._cells, t, oid)
        i = int(np.argmax(p))
        if p[i] < self.threshold:
            return None
        return (self._cells[i].copy(), self.last_seen[oid], float(p[i]))

    def query_at(self, object_id: int, t: int) -> Optional[np.ndarray]:
        r = self.query_current(object_id, t)
        return None if r is None else r[0]

    def query_region(self, bounds, t: int) -> List[int]:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        mask = np.all((self._cells >= lo) & (self._cells <= hi), axis=1)
        cells = self._cells[mask]
        if len(cells) == 0:
            return []
        out = []
        for oid in sorted(self.last_seen):
            p = self._occupancy(cells, t, oid)
            if float(p.max()) >= self.threshold:
                out.append(oid)
        return out

    def stale_ids(self, t: int, max_age: int) -> List[int]:
        cutoff = int(t) - int(max_age)
        return sorted(oid for oid, s in self.last_seen.items() if s <= cutoff)

    def size_bytes(self) -> int:
        weights = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        if self.depth == 2:
            weights += self.W1b.size + self.b1b.size
        return int(8 * weights + 48 * len(self.buffer) + 24 * len(self.last_seen))
