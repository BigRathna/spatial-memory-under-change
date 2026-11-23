# Spatial memory under environmental change

A small, CPU-only benchmark comparing three ways for an embodied agent to
remember where things are while the world changes underneath it.

The question:

> As an environment changes and the mission gets longer, which memory
> representation stays correct at an acceptable memory and update cost?

Three memories share one interface:

- an object graph,
- a dense 16 x 16 x 8 voxel grid,
- a 32-unit coordinate MLP trained online in numpy.

A synthetic world supplies observations with detection noise and occlusion.
Objects move, appear and are removed. Queries are drawn from ground truth:
current position, past position, region change, and stale objects.

## Results

Mean over three seeds. Full table and discussion in [REPORT.md](REPORT.md).

| Memory | Current success@2 | Past success@2 | Persistent size | Peak memory | Update |
|---|---|---|---|---|---|
| Object graph | 0.880 | 0.867 | 15.4 KB, growing | 63 KB | 0.016 ms |
| Dense grid | 0.880 | 0.693 | 24.6 KB, flat | 31 KB | 0.142 ms |
| Coordinate MLP | 0.027 | 0.013 | 12.2 KB | 1.36 MB | 40.7 ms |

The graph keeps identity and history but grows. The dense grid is flat but
loses history and ids to voxel collisions. The small MLP is compact on disk and
expensive in RAM, and this configuration has no usable operating point. That is
a result about one 32-unit, 50-step model, not about implicit representations as
a class.

The report also computes the overhead ratio `peak memory / persistent size`:
4.12 for the graph, 1.27 for the dense grid, and 111.9 for the MLP.

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy matplotlib pytest
./run.sh
.venv/bin/python -m pytest -q
```

`./run.sh` runs three seeds in about nine seconds on CPU and writes
`results/metrics.csv`, `results/metrics.json`, `results/table.md` and
`results/figure.png`.

The diagnostics sweep takes about 35 seconds:

```bash
.venv/bin/python -m src.diagnostics --seeds 3
```

## Files

| Path | Contents |
|---|---|
| `src/world.py` | Grid world, sensing with noise, occlusion and detection probability, change process |
| `src/memories.py` | Object graph, dense grid and coordinate MLP behind one `Memory` interface |
| `src/metrics.py` | Query sampling and metric computation |
| `src/demo.py` | Runs seeds, evaluates the memories, writes CSV, JSON, table and figure |
| `src/diagnostics.py` | Object-count sweep behind the failure case |
| `tests/` | 31 tests, including the original 21 and the extension suite |
| `results/` | Generated outputs |
| `REPORT.md` | Two-page report |
| `IMPLICIT_MEMORY_NOTES.md` | Candidate mechanisms for the MLP failure, kept out of the report |
| `EXTENSIONS.md` | Capacity sweep, relational queries, eviction and a 300-step run |

## Extensions

`EXTENSIONS.md` adds three follow-ups to the original benchmark:

- **Capacity.** Six MLP width/depth settings at 3 and 30 objects. No setting
  exceeds 0.184 current success at 3 objects or 0.024 at 30; the graph holds
  0.896 and 0.904.
- **Relational queries.** Left, right, near and nearest-of-class. Graph 0.85 to
  0.94, dense 0.79 to 0.86, MLP 0.20 to 0.43.
- **Maintenance and horizon.** Dense eviction does not reduce the 24.6 KB map;
  age-10 eviction cuts past success to 0.112. A 300-step run grows the graph
  from 16.8 to 88.4 KB with query latency 0.158 to 0.228 ms.

## Scope

Synthetic world, CPU only, no downloaded data, no GPU. The numbers describe the
representations, not a deployment. Persistent size is an estimate of serialized
state, not live Python object overhead.

## License

MIT. See [LICENSE](LICENSE).
