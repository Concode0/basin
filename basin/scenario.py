"""Configuration and a fixed sparse heterogeneous graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from .model import Node


@dataclass(frozen=True)
class Workload:
    arrival_attempts: int = 3
    arrival_probability: float = 0.6
    compute_work_range: tuple[int, int] = (12, 48)
    memory_required_range: tuple[int, int] = (2, 12)


@dataclass(frozen=True)
class Scenario:
    nodes: tuple[Node, ...]
    seed: int = 7
    gossip_delay: int = 3
    task_hop_delay: int = 3
    gossip_period: int = 2
    packet_loss: float = 0.04
    max_hops: int = 7
    workload: Workload = field(default_factory=Workload)


def default_scenario(seed: int = 7) -> Scenario:
    count = 20
    ids = tuple(f"N{i:02d}" for i in range(count))
    # A cycle plus opposite-node links: connected, deterministic, degree three.
    nodes = tuple(Node(node_id, 1.5 + ((i * 7) % 17) / 3,
                       4 + ((i * 11 + i * i) % 11),
                       tuple(sorted({ids[(i - 1) % count], ids[(i + 1) % count],
                                     ids[(i + count // 2) % count]})))
                  for i, node_id in enumerate(ids))
    return Scenario(nodes=nodes, seed=seed)
