"""Small values passed across the simulator's locality boundary."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, TypeAlias

@dataclass(frozen=True)
class Node:
    id: str
    cpu_rate: float
    memory_capacity: float
    neighbors: tuple[str, ...]

@dataclass(frozen=True)
class Task:
    task_id: str
    origin: str
    created_tick: int
    compute_work: float
    remaining_work: float
    memory_required: float
    status: str = "queued"
    current_node: str | None = None
    source: str | None = None
    target: str | None = None
    depart_tick: int | None = None
    arrive_tick: int | None = None
    hops: int = 0
    started_tick: int | None = None
    completed_tick: int | None = None

@dataclass(frozen=True)
class BeliefRecord:
    node_id: str
    cpu_rate: float
    memory_capacity: float
    known_remaining_work: float
    version: int
    observation_tick: int

@dataclass(frozen=True)
class LocalView:
    node_id: str
    self_state: BeliefRecord
    neighbors: tuple[str, ...]
    belief: Mapping[str, BeliefRecord]
    tick: int

@dataclass(frozen=True)
class Accept:
    pass

@dataclass(frozen=True)
class Hold:
    pass

@dataclass(frozen=True)
class Forward:
    target_node: str

Action: TypeAlias = Accept | Hold | Forward
