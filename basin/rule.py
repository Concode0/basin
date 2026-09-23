"""A local quadratic-energy balancing rule."""
from __future__ import annotations
from typing import Protocol
from .model import Accept, Action, Forward, Hold, LocalView, Task


class Rule(Protocol):
    def decide(self, task: Task, view: LocalView) -> Action: ...


def quadratic_gain(own_work: float, own_rate: float, peer_work: float,
                   peer_rate: float, task_work: float) -> float:
    before = own_work**2 / (2 * own_rate) + peer_work**2 / (2 * peer_rate)
    after = (own_work - task_work)**2 / (2 * own_rate) + (peer_work + task_work)**2 / (2 * peer_rate)
    return before - after


class BasinRule:
    """Forward the pending task only for the best positive local gain."""
    epsilon = 1e-9

    def __init__(self, max_hops: int = 7):
        self.max_hops = max_hops

    def decide(self, task: Task, view: LocalView) -> Action:
        own = view.self_state
        if task.hops < self.max_hops:
            best_id = None
            best_gain = self.epsilon
            for neighbor in view.neighbors:
                peer = view.belief.get(neighbor)
                if peer is None or task.memory_required > peer.memory_capacity:
                    continue
                gain = quadratic_gain(own.known_remaining_work, own.cpu_rate,
                                      peer.known_remaining_work, peer.cpu_rate,
                                      task.remaining_work)
                if gain > best_gain:
                    best_gain, best_id = gain, neighbor
            if best_id is not None:
                return Forward(best_id)
        return Accept() if task.memory_required <= own.memory_capacity else Hold()
