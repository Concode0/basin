import copy
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from PIL import Image

from basin.model import Forward, Hold, Node
from basin.scenario import Scenario, Workload
from basin.simulator import Simulator
from basin.viewer import ReplayViewer, save_gif


class ForwardFromA:
    def decide(self, task, view):
        return Forward("B") if view.node_id == "A" else Hold()


def frames():
    scenario = Scenario(nodes=(Node("A", 2, 8, ("B",)), Node("B", 4, 8, ("A",))),
                        seed=4, packet_loss=0, gossip_delay=3, task_hop_delay=3,
                        gossip_period=1, workload=Workload(1, 1, (8, 8), (2, 2)))
    return Simulator(scenario, ForwardFromA()).run(5)


class ViewerTests(unittest.TestCase):
    def test_subframe_moves_only_transit_markers_and_keeps_normalization(self):
        shots = frames()
        original = copy.deepcopy(shots)
        viewer = ReplayViewer(shots)
        try:
            norm = viewer.norm
            viewer.render(1, 0.0)
            start = viewer._task_hit["T0000"]
            discrete_artists = tuple(viewer.ax.collections)
            viewer.render(1, 0.5)
            halfway = viewer._task_hit["T0000"]
            self.assertEqual(halfway, viewer._between(viewer.positions, "A", "B", 1, 4, 1.5))
            self.assertNotEqual(start, halfway)
            self.assertEqual(discrete_artists, tuple(viewer.ax.collections))
            self.assertIs(norm, viewer.norm)
            viewer.set_frame(2)
            self.assertEqual(viewer.subframe, 0)
            self.assertIs(norm, viewer.norm)
            self.assertEqual(shots, original)
        finally:
            matplotlib.pyplot.close(viewer.fig)

    def test_gif_export_is_reproducible_and_does_not_mutate_frames(self):
        shots = frames()[:3]
        original = copy.deepcopy(shots)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.gif"
            second = Path(directory) / "second.gif"
            save_gif(shots, first, fps=20)
            save_gif(shots, second, fps=20)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with Image.open(first) as gif:
                self.assertGreater(gif.n_frames, len(shots))
        self.assertEqual(shots, original)


if __name__ == "__main__":
    unittest.main()
