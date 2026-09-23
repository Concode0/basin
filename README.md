# Basin

A small experiment: can sparse, heterogeneous nodes spread randomly arriving work toward a moving balance using only local decisions and delayed gossip?

## Setup

Twenty fixed nodes have different CPU rates and memory capacities. Each has three graph neighbors. Tasks arrive at random origins with random compute work and memory needs; every task fits its origin or a direct neighbor. One worker per node consumes `cpu_rate` units of work per tick. Nodes know their own work exactly, while versioned reports from neighbors arrive after three ticks and can be lost.

## Rule

A node's pressure is `p_i = W_i / c_i`, where `W_i` is its queued and running work and `c_i` is its CPU rate. The experiment tracks `Phi = sum(W_i² / (2 c_i))`. For a pending task of work `w`, the node estimates how moving it to each memory-feasible direct neighbor would change its two-node contribution to `Phi`. It forwards to the neighbor with the largest positive decrease; otherwise it accepts the task locally. An origin that cannot execute the task waits if no positive feasible move is known yet. A hop limit is the final safety bound. The rule sees exact local work and gossiped neighbor records, never global truth.

## What I observed

With 19 initially imbalanced tasks and no later arrivals, refreshing neighbor records just before decisions gave 53 forwards in 180 ticks; every move had positive true quadratic gain (smallest `+7.95`). By tick 24, placement had settled with work still running and no positive feasible one-hop move among the remaining runnable tasks. With the normal three-tick gossip delay and 4% packet loss, the same initial workload made 35 forwards. One had negative true gain (`-156.6` at tick 10); that task moved again as information changed. The snapshot potential can also jump when work in transit arrives, since `W_i` counts only queued and running work. Per-move gain is the direct check of the rule's energy condition.

The default arrival rate offers about 54 work units per tick against 82.3 units of aggregate CPU capacity, about 66%. In a 5,000-tick seed-7 run, mean outstanding work over successive 500-tick windows ranged from 850 to 1,212 units and ended at 927. It fluctuated rather than draining or rising steadily. This is a local, dynamic balance, not a guaranteed global optimum: tasks are indivisible, edges are sparse, memory is only a hard constraint, and gossip is delayed.

## Visualization

The Matplotlib map fixes each node at `(cpu_rate, memory_capacity)` and colors its sampled pressure. Contours interpolate those discrete samples for viewing only; the field and geometric distances never affect routing. Edges show communication links, while task and gossip markers move along their actual transits. Playback uses discrete simulation snapshots with smooth visual transit positions. Click nodes to inspect beliefs and tasks to inspect projected pressure. [View a 100-tick GIF](basin.gif).

## Running

```bash
uv sync
uv run python -m basin
uv run python -m basin --ticks 300 --seed 7 --gif basin.gif
uv run python -m unittest discover -s tests -v
```

Use `--gif-only` to export without opening the viewer, or `--fps` to change visual playback and GIF rate. Code can call `basin.viewer.save_gif(frames, path, fps=20)` directly.
