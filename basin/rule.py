"""The baseline local scheduling rule."""
from __future__ import annotations
from typing import Protocol
from .model import Accept, Action, BeliefRecord, Forward, Hold, LocalView, Task

class Rule(Protocol):
    def decide(self, task: Task, view: LocalView) -> Action: ...

class BasinRule:
    def __init__(self, max_hops: int = 7, hysteresis: float = 0.5):
        self.max_hops = max_hops
        self.hysteresis = hysteresis

    @staticmethod
    def pressure(task: Task, record: BeliefRecord) -> float | None:
        if task.memory_required > record.memory_capacity:
            return None
        return (record.known_remaining_work + task.compute_work) / record.cpu_rate

    def decide(self, task: Task, view: LocalView) -> Action:
        own = self.pressure(task, view.self_state)
        if task.hops >= self.max_hops:
            return Accept() if own is not None else Hold()
        best_id = None
        best = float("inf")
        for neighbor in view.neighbors:
            record = view.belief.get(neighbor)
            if record is None:
                continue
            score = self.pressure(task, record)
            if score is not None and (score, neighbor) < (best, best_id or ""):
                best, best_id = score, neighbor
        if best_id is not None and (own is None or best + self.hysteresis < own):
            return Forward(best_id)
        return Accept() if own is not None else Hold()
