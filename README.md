# Basin

Basin is a small deterministic sandbox for studying decentralized scheduling under heterogeneous resources and delayed local knowledge. It asks whether heterogeneous nodes can distribute randomly arriving work through local decisions and delayed gossip.

## Run

```bash
uv sync
uv run python -m basin
uv run python -m basin --ticks 300 --seed 7
uv run python -m basin --ticks 300 --seed 7 --gif basin.gif
uv run python -m basin --ticks 300 --seed 7 --gif basin.gif --gif-only
```

The command precomputes snapshots and opens one interactive Matplotlib figure. Playback uses a 350 ms simulation tick and about 20 visual frames per second; only in-flight marker positions move between snapshots. Use Play/Pause, Prev/Next, or the slider to inspect discrete frames. Left/Right and Space also work. Click a node to show its belief; click it again for global truth. Click an active task for its task-specific projected pressure; click again to clear. Escape clears both selections. `--gif` exports through Matplotlib and Pillow before opening the viewer; `--fps` changes visual playback and export rate. Python code can call `basin.viewer.save_gif(frames, path, fps=20)` directly.

## Experiment

Nodes are fixed heterogeneous machines with a CPU rate, memory capacity, and communication neighbors. A node has one worker slot. Seeded random arrivals choose an origin, compute work, and memory requirement. Every generated task fits at least one machine, though its origin may be infeasible. A running task loses `cpu_rate` units of remaining work per tick. Memory is a hard feasibility constraint, not a changing coordinate or shared allocation model.

Each tick delivers due messages and task transits, advances running work, starts one runnable task per idle node, lets each node decide on at most one pending task, applies forwarding, injects arrivals, schedules gossip, and stores a detached snapshot. New tasks therefore appear before their first placement decision. Forwarded tasks and gossip travel for multiple ticks. Versioned state reports move over graph edges, with optional seeded packet loss. A node always knows its own work exactly, while remote reports can be old or unknown.

`Rule.decide(task, LocalView)` receives a frozen task and only local state, neighbor IDs, and local belief records. The initial `BasinRule` filters infeasible candidates, compares `(known_remaining_work + task.compute_work) / cpu_rate`, and forwards only when a known neighbor improves projected pressure by more than a small hysteresis. An infeasible origin forwards to a known feasible neighbor when possible. A hop limit bounds routing. This is a baseline rule to replace and test, not a claim of a novel scheduler.

## Map

Every node stays at `(cpu_rate, memory_capacity)`. Thin lines show actual communication edges; geometric length has no latency meaning. The ordinary potential map interpolates discrete `remaining_work / cpu_rate` samples using one stable color range across the replay. Selecting a task shows projected pressure and marks memory-infeasible nodes with X markers. Selecting a node reconstructs the same map from that node's current belief, leaving unknown nodes hollow. Yellow markers show tasks; orange marks running work; small blue markers show in-flight gossip. Transit positions come directly from departure, fractional display time, and arrival ticks. Pressure, beliefs, queues, and task lifecycle change only at snapshot boundaries.

The interpolated field is a human visualization of discrete node costs. It is not part of routing semantics or a physical simulation. Scheduling uses graph neighbors and belief records only. The useful observations are stale decisions, oscillations, queue growth, and imbalance as information catches up.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The repository has no server, browser viewer, or web replay layer. Git history retains the previous implementations.
