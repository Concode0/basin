import dataclasses
import unittest
from basin.model import Accept, BeliefRecord, Forward, Hold, LocalView, Node, Task
from basin.rule import BasinRule
from basin.scenario import Scenario, Workload, default_scenario
from basin.simulator import Simulator, merge_record


def graph(nodes=None, **changes):
    values = dict(nodes=nodes or (Node("A", 2, 4, ("B",)), Node("B", 4, 8, ("A",))),
                  seed=4, packet_loss=0, gossip_delay=3, task_hop_delay=3,
                  gossip_period=1, workload=Workload(arrival_attempts=0))
    values.update(changes)
    return Scenario(**values)


class HoldRule:
    def __init__(self):
        self.calls = []
    def decide(self, task, view):
        self.calls.append((task, view))
        return Hold()


class AcceptRule:
    def decide(self, task, view):
        return Accept()


class ForwardRule:
    def decide(self, task, view):
        return Forward("B") if view.node_id == "A" else Hold()


class SimulatorTests(unittest.TestCase):
    def test_seed_and_stochastic_generation(self):
        scenario = default_scenario(59)
        self.assertEqual(Simulator(scenario).run(50), Simulator(scenario).run(50))
        hold = HoldRule()
        shots = Simulator(scenario, hold).run(100)
        tasks = shots[-1]["tasks"].values()
        self.assertGreater(len({t["origin"] for t in tasks}), 10)
        self.assertTrue(all(any(t["memory_required"] <= n.memory_capacity for n in scenario.nodes) for t in tasks))
        self.assertTrue(any(t["memory_required"] > next(n.memory_capacity for n in scenario.nodes if n.id == t["origin"])
                            for t in tasks))
        self.assertTrue(any(not any(t["created_tick"] == tick for t in tasks) for tick in range(100)))
        self.assertEqual(shots[0], Simulator(scenario, hold).step())

    def test_rule_locality_and_one_decision_per_tick(self):
        rule = HoldRule()
        scenario = graph(nodes=(Node("A", 2, 8, ()),),
                         workload=Workload(3, 1, (5, 5), (1, 1)))
        sim = Simulator(scenario, rule)
        first = sim.step()
        self.assertEqual(len(first["nodes"]["A"]["pending"]), 3)
        self.assertEqual(len(rule.calls), 0)
        second = sim.step()
        self.assertEqual(len(rule.calls), 1)
        self.assertEqual(len(second["nodes"]["A"]["pending"]), 6)
        task, view = rule.calls[0]
        self.assertIsInstance(task, Task)
        self.assertIsInstance(view, LocalView)
        self.assertEqual(set(vars(view)), {"node_id", "self_state", "neighbors", "belief", "tick"})
        with self.assertRaises(TypeError):
            view.belief["B"] = view.self_state
        with self.assertRaises(dataclasses.FrozenInstanceError):
            task.status = "completed"
        self.assertEqual(view.self_state.known_remaining_work, 10)  # other two pending tasks
        self.assertEqual(view.self_state.node_id, "A")

    def test_rule_feasibility_backlog_hysteresis_and_stale_view(self):
        task = Task("T", "A", 0, 10, 10, 6, current_node="A")
        a = BeliefRecord("A", 2, 8, 10, 1, 0)
        b = BeliefRecord("B", 4, 8, 0, 1, 0)
        view = LocalView("A", a, ("B",), {"B": b}, 1)
        self.assertEqual(BasinRule().decide(task, view), Forward("B"))
        self.assertEqual(BasinRule(hysteresis=8).decide(task, view), Accept())
        self.assertEqual(BasinRule(max_hops=0).decide(task, view), Accept())
        self.assertEqual(BasinRule().decide(task, LocalView("A", a, ("B",), {}, 1)), Accept())
        impossible_here = dataclasses.replace(a, memory_capacity=4)
        self.assertEqual(BasinRule().decide(task, LocalView("A", impossible_here, ("B",), {"B": b}, 1)), Forward("B"))
        impossible_there = dataclasses.replace(b, memory_capacity=4)
        self.assertEqual(BasinRule().decide(task, LocalView("A", a, ("B",), {"B": impossible_there}, 1)), Accept())
        stale = dataclasses.replace(b, known_remaining_work=100)
        truth = dataclasses.replace(b, known_remaining_work=0)
        self.assertEqual(BasinRule().decide(task, LocalView("A", a, ("B",), {"B": stale}, 5)), Accept())
        self.assertEqual(BasinRule().decide(task, LocalView("A", a, ("B",), {"B": truth}, 5)), Forward("B"))

    def test_task_transit_and_gossip_delays(self):
        w = Workload(1, 1, (8, 8), (2, 2))
        sim = Simulator(graph(workload=w), ForwardRule())
        shots = sim.run(5)
        self.assertEqual(shots[0]["tasks"]["T0000"]["status"], "queued")
        self.assertEqual(shots[1]["task_transits"][0]["arrive_tick"], 4)
        self.assertEqual((shots[1]["tasks"]["T0000"]["source"],
                          shots[1]["tasks"]["T0000"]["target"],
                          shots[1]["tasks"]["T0000"]["depart_tick"],
                          shots[1]["tasks"]["T0000"]["arrive_tick"]), ("A", "B", 1, 4))
        for tick in (1, 2, 3):
            self.assertEqual(shots[tick]["tasks"]["T0000"]["status"], "in_transit")
        self.assertEqual(shots[4]["tasks"]["T0000"]["current_node"], "B")
        self.assertEqual(shots[0]["gossip_messages"][0]["arrive_tick"], 3)
        self.assertNotIn("A", shots[2]["beliefs"]["B"])
        self.assertIn("A", shots[3]["beliefs"]["B"])
        lost = Simulator(graph(packet_loss=1)).run(8)
        self.assertEqual(set(lost[-1]["beliefs"]["B"]), {"B"})

    def test_newer_record_wins(self):
        base = BeliefRecord("B", 4, 8, 5, 2, 1)
        belief = {"B": base}
        self.assertFalse(merge_record(belief, dataclasses.replace(base, version=1), "A"))
        self.assertTrue(merge_record(belief, dataclasses.replace(base, version=3), "A"))
        self.assertFalse(merge_record(belief, dataclasses.replace(base, node_id="A", version=99), "A"))

    def test_execution_and_completion(self):
        scenario = graph(nodes=(Node("A", 3, 8, ()),),
                         workload=Workload(1, 1, (7, 7), (2, 2)))
        shots = Simulator(scenario, AcceptRule()).run(6)
        t = lambda i: shots[i]["tasks"]["T0000"]
        self.assertEqual(t(0)["status"], "queued")
        self.assertEqual(t(1)["status"], "queued")  # accepted, starts next tick
        self.assertEqual((t(2)["status"], t(2)["remaining_work"]), ("running", 7))
        self.assertEqual(t(3)["remaining_work"], 4)
        self.assertEqual(t(4)["remaining_work"], 1)
        self.assertEqual((t(5)["status"], t(5)["remaining_work"]), ("completed", 0))
        self.assertIsNone(t(5)["current_node"])
        self.assertNotEqual(shots[5]["nodes"]["A"]["running"], "T0000")
        self.assertNotIn("T0000", shots[5]["nodes"]["A"]["runnable"])

if __name__ == "__main__":
    unittest.main()
