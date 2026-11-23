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
| `tests/` | 21 tests, including a report-table consistency check |
| `results/` | Generated outputs |
| `REPORT.md` | Two-page report |
| `IMPLICIT_MEMORY_NOTES.md` | Candidate mechanisms for the MLP failure, kept out of the report |

## Scope

Synthetic world, CPU only, no downloaded data, no GPU. The numbers describe the
representations, not a deployment. Persistent size is an estimate of serialized
state, not live Python object overhead.

## License

MIT. See [LICENSE](LICENSE).
