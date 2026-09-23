"""Configuration and a fixed heterogeneous graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from .model import Node

@dataclass(frozen=True)
class Workload:
    arrival_attempts: int = 2
    arrival_probability: float = 0.55
    compute_work_range: tuple[int, int] = (12, 48)
    memory_required_range: tuple[int, int] = (2, 14)

@dataclass(frozen=True)
class Scenario:
    nodes: tuple[Node, ...]
    seed: int = 7
    gossip_delay: int = 3
    task_hop_delay: int = 3
    gossip_period: int = 2
    packet_loss: float = 0.04
    max_hops: int = 7
    hysteresis: float = 0.5
    workload: Workload = field(default_factory=Workload)

def default_scenario(seed: int = 7) -> Scenario:
    count = 20
    ids = tuple(f"N{i:02d}" for i in range(count))
    links = {node_id: set() for node_id in ids}
    for i in range(count):
        for j in ((i + 1) % count, (i + 5) % count if i % 4 == 0 else (i + 1) % count):
            links[ids[i]].add(ids[j])
            links[ids[j]].add(ids[i])
    nodes = tuple(Node(node_id, 1.5 + ((i * 7) % 17) / 3,
                       4 + ((i * 11 + i * i) % 11), tuple(sorted(links[node_id])))
                  for i, node_id in enumerate(ids))
    return Scenario(nodes=nodes, seed=seed)
