"""Single-threaded, deterministic scheduling and delayed graph messages."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from random import Random
from types import MappingProxyType
from .model import Accept, BeliefRecord, Forward, Hold, LocalView, Node, Task
from .rule import BasinRule, Rule, quadratic_gain
from .scenario import Scenario

@dataclass
class _NodeState:
    hardware: Node
    pending: list[str] = field(default_factory=list)
    runnable: list[str] = field(default_factory=list)
    running: str | None = None
    belief: dict[str, BeliefRecord] = field(default_factory=dict)
    version: int = 0

@dataclass(frozen=True)
class TaskTransit:
    task_id: str
    source: str
    target: str
    depart_tick: int
    arrive_tick: int

@dataclass(frozen=True)
class GossipMessage:
    source: str
    target: str
    depart_tick: int
    arrive_tick: int
    records: tuple[BeliefRecord, ...]

def merge_record(belief: dict[str, BeliefRecord], record: BeliefRecord, own_id: str) -> bool:
    if record.node_id == own_id:
        return False
    old = belief.get(record.node_id)
    if old is None or record.version > old.version:
        belief[record.node_id] = record
        return True
    return False

class Simulator:
    """Owns global truth; Rule sees only a frozen task and LocalView."""
    def __init__(self, scenario: Scenario, rule: Rule | None = None):
        self._validate(scenario)
        self.scenario = scenario
        self.rule = rule if rule is not None else BasinRule(scenario.max_hops)
        self.rng = Random(scenario.seed)
        self._nodes = {n.id: _NodeState(n) for n in sorted(scenario.nodes, key=lambda n: n.id)}
        self._tasks: dict[str, Task] = {}
        self._task_transits: list[TaskTransit] = []
        self._gossip_messages: list[GossipMessage] = []
        self.tick_index = 0
        self._next_task = 0
        self._forward_count = 0
        self._decisions: list[dict] = []
        for state in self._nodes.values():
            self._report(state, -1)

    @staticmethod
    def _validate(s: Scenario) -> None:
        nodes = {n.id: n for n in s.nodes}
        if not nodes or len(nodes) != len(s.nodes):
            raise ValueError("nodes must have unique ids")
        if any(n.cpu_rate <= 0 or n.memory_capacity <= 0 or len(set(n.neighbors)) != len(n.neighbors)
               or n.id in n.neighbors or any(peer not in nodes for peer in n.neighbors) for n in s.nodes):
            raise ValueError("invalid node hardware or topology")
        if any(n.id not in nodes[peer].neighbors for n in s.nodes for peer in n.neighbors):
            raise ValueError("topology must be undirected")
        w = s.workload
        if (s.gossip_delay < 2 or s.task_hop_delay < 2 or s.gossip_period < 1
            or s.max_hops < 0 or not 0 <= s.packet_loss <= 1
            or w.arrival_attempts < 0 or not 0 <= w.arrival_probability <= 1
            or w.compute_work_range[0] <= 0 or w.compute_work_range[1] < w.compute_work_range[0]
            or w.memory_required_range[0] <= 0 or w.memory_required_range[1] < w.memory_required_range[0]
            or any(w.memory_required_range[0] > max([n.memory_capacity] +
                   [nodes[peer].memory_capacity for peer in n.neighbors]) for n in s.nodes)):
            raise ValueError("invalid simulation configuration")

    def _remaining(self, state: _NodeState) -> float:
        ids = state.pending + state.runnable + ([state.running] if state.running else [])
        return sum(self._tasks[task_id].remaining_work for task_id in ids)

    def _report(self, state: _NodeState, tick: int) -> None:
        state.version += 1
        n = state.hardware
        state.belief[n.id] = BeliefRecord(n.id, n.cpu_rate, n.memory_capacity,
                                           self._remaining(state), state.version, tick)

    def local_view(self, node_id: str, task_id: str | None = None) -> LocalView:
        state = self._nodes[node_id]
        own = state.belief[node_id]
        if task_id is not None:
            if task_id not in state.pending:
                raise ValueError("task is not pending locally")
        return LocalView(node_id, own, state.hardware.neighbors,
                         MappingProxyType(dict(state.belief)), self.tick_index)

    def step(self) -> dict:
        tick = self.tick_index
        forwards_before = self._forward_count
        self._deliver(tick)
        self._advance(tick)
        self._start(tick)
        decisions = []
        self._decisions = []
        for node_id, state in self._nodes.items():
            if not state.pending:
                continue
            task_id = state.pending[0]
            view = self.local_view(node_id, task_id)
            action = self.rule.decide(self._tasks[task_id], view)
            decisions.append((node_id, task_id, action))
            target = action.target_node if isinstance(action, Forward) else None
            actual_gain = None
            if target in self._nodes:
                source_state = self._nodes[node_id]
                target_state = self._nodes[target]
                actual_gain = quadratic_gain(self._remaining(source_state), source_state.hardware.cpu_rate,
                                             self._remaining(target_state), target_state.hardware.cpu_rate,
                                             self._tasks[task_id].remaining_work)
            self._decisions.append({"node_id": node_id, "task_id": task_id,
                                    "action": type(action).__name__.lower(),
                                    "target_node": target, "actual_gain": actual_gain})
        for node_id, task_id, action in decisions:
            self._apply(node_id, task_id, action, tick)
        self._generate(tick)
        self._schedule_gossip(tick)
        shot = self.snapshot(tick)
        shot["diagnostics"]["forwarded_this_tick"] = self._forward_count - forwards_before
        self.tick_index += 1
        return shot

    def run(self, ticks: int) -> list[dict]:
        if ticks < 0:
            raise ValueError("ticks must be nonnegative")
        return [self.step() for _ in range(ticks)]

    def _deliver(self, tick: int) -> None:
        waiting = []
        for message in self._gossip_messages:
            if message.arrive_tick > tick:
                waiting.append(message)
            else:
                receiver = self._nodes[message.target]
                for record in message.records:
                    merge_record(receiver.belief, record, message.target)
        self._gossip_messages = waiting
        waiting_tasks = []
        for transit in self._task_transits:
            if transit.arrive_tick > tick:
                waiting_tasks.append(transit)
            else:
                state = self._nodes[transit.target]
                task = self._tasks[transit.task_id]
                self._tasks[transit.task_id] = replace(task, status="queued", current_node=transit.target,
                                                        source=None, target=None, depart_tick=None, arrive_tick=None)
                state.pending.append(transit.task_id)
                self._report(state, tick)
        self._task_transits = waiting_tasks

    def _advance(self, tick: int) -> None:
        for state in self._nodes.values():
            if state.running is None:
                continue
            task = self._tasks[state.running]
            remaining = max(0.0, task.remaining_work - state.hardware.cpu_rate)
            if remaining == 0:
                self._tasks[task.task_id] = replace(task, remaining_work=0, status="completed",
                                                    current_node=None, completed_tick=tick)
                state.running = None
            else:
                self._tasks[task.task_id] = replace(task, remaining_work=remaining)
            self._report(state, tick)

    def _start(self, tick: int) -> None:
        for state in self._nodes.values():
            if state.running is not None or not state.runnable:
                continue
            task_id = state.runnable.pop(0)
            task = self._tasks[task_id]
            if task.memory_required > state.hardware.memory_capacity:
                raise RuntimeError("infeasible task entered runnable queue")
            state.running = task_id
            self._tasks[task_id] = replace(task, status="running", started_tick=tick)
            self._report(state, tick)

    def _apply(self, node_id: str, task_id: str, action: object, tick: int) -> None:
        state = self._nodes[node_id]
        task = self._tasks[task_id]
        if not state.pending or state.pending[0] != task_id:
            raise RuntimeError("decision does not refer to pending front")
        if isinstance(action, Hold):
            return
        if isinstance(action, Accept):
            if task.memory_required > state.hardware.memory_capacity:
                raise ValueError("rule accepted an infeasible task")
            state.pending.pop(0)
            state.runnable.append(task_id)
            self._report(state, tick)
            return
        if isinstance(action, Forward):
            if action.target_node not in state.hardware.neighbors or task.hops >= self.scenario.max_hops:
                raise ValueError("rule forwarded outside topology or hop limit")
            state.pending.pop(0)
            arrive = tick + self.scenario.task_hop_delay
            self._tasks[task_id] = replace(task, status="in_transit", current_node=None,
                                            source=node_id, target=action.target_node,
                                            depart_tick=tick, arrive_tick=arrive, hops=task.hops + 1)
            self._task_transits.append(TaskTransit(task_id, node_id, action.target_node, tick, arrive))
            self._forward_count += 1
            self._report(state, tick)
            return
        raise TypeError(f"unknown action: {action!r}")

    def _generate(self, tick: int) -> None:
        w = self.scenario.workload
        ids = tuple(self._nodes)
        for _ in range(w.arrival_attempts):
            if self.rng.random() >= w.arrival_probability:
                continue
            origin = self.rng.choice(ids)
            work = self.rng.randint(*w.compute_work_range)
            state = self._nodes[origin]
            local_capacity = max([state.hardware.memory_capacity] +
                                 [self._nodes[peer].hardware.memory_capacity for peer in state.hardware.neighbors])
            maximum = min(w.memory_required_range[1], int(local_capacity))
            memory = self.rng.randint(w.memory_required_range[0], maximum)
            task_id = f"T{self._next_task:04d}"
            self._next_task += 1
            self._tasks[task_id] = Task(task_id, origin, tick, work, work, memory, current_node=origin)
            state.pending.append(task_id)
            self._report(state, tick)

    def _schedule_gossip(self, tick: int) -> None:
        if tick % self.scenario.gossip_period:
            return
        for state in self._nodes.values():
            records = tuple(state.belief[key] for key in sorted(state.belief))
            for peer in state.hardware.neighbors:
                if self.rng.random() >= self.scenario.packet_loss:
                    self._gossip_messages.append(GossipMessage(state.hardware.id, peer, tick,
                                                                tick + self.scenario.gossip_delay, records))

    def snapshot(self, tick: int | None = None) -> dict:
        if tick is None:
            tick = self.tick_index - 1
        nodes = {}
        for node_id, state in self._nodes.items():
            nodes[node_id] = {**asdict(state.hardware), "remaining_work": self._remaining(state),
                              "pending": tuple(state.pending), "runnable": tuple(state.runnable),
                              "running": state.running}
        pressures = [node["remaining_work"] / node["cpu_rate"] for node in nodes.values()]
        diagnostics = {
            "phi": sum(node["remaining_work"]**2 / (2 * node["cpu_rate"]) for node in nodes.values()),
            "total_remaining_work": sum(task.remaining_work for task in self._tasks.values()
                                        if task.status != "completed"),
            "completed_count": sum(task.status == "completed" for task in self._tasks.values()),
            "mean_pressure": sum(pressures) / len(pressures),
            "max_pressure": max(pressures),
            "forwarded_count": self._forward_count,
            "forwarded_this_tick": 0,
        }
        return {"tick": tick, "nodes": nodes,
                "tasks": {key: asdict(task) for key, task in self._tasks.items()},
                "beliefs": {node_id: {key: asdict(record) for key, record in state.belief.items()}
                            for node_id, state in self._nodes.items()},
                "task_transits": [asdict(t) for t in self._task_transits],
                "gossip_messages": [asdict(m) for m in self._gossip_messages],
                "decisions": [dict(decision) for decision in self._decisions],
                "diagnostics": diagnostics}
