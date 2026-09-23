"""One Matplotlib figure for deterministic snapshot replay and GIF export."""
from __future__ import annotations
import math
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider
from PIL import GifImagePlugin, Image

TICK_SECONDS = 0.35
DEFAULT_FPS = 20


class ReplayViewer:
    def __init__(self, frames: list[dict], fps: int = DEFAULT_FPS, controls: bool = True):
        if not frames:
            raise ValueError("replay needs at least one frame")
        if fps < 1:
            raise ValueError("fps must be positive")
        self.frames = frames
        self.fps = fps
        self.subframes = max(1, round(TICK_SECONDS * fps))
        self.index = 0
        self.subframe = 0
        self.playing = False
        self.selected_node: str | None = None
        self.selected_task: str | None = None
        self.positions = {key: (node["cpu_rate"], node["memory_capacity"])
                          for key, node in frames[0]["nodes"].items()}
        self._task_hit: dict[str, tuple[float, float]] = {}
        self._static_task_hit: dict[str, tuple[float, float]] = {}
        self._rendered_index: int | None = None
        self._rendered_selection = None
        self._updating_slider = False
        self.cmap = plt.get_cmap("viridis")
        maximum = max(node["remaining_work"] / node["cpu_rate"]
                      for frame in frames for node in frame["nodes"].values())
        self.truth_norm = Normalize(0, max(1.0, maximum), clip=True)
        self.norm = self.truth_norm
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(bottom=0.16 if controls else 0.10)
        self.slider = None
        self.timer = None
        if controls:
            previous_ax = self.fig.add_axes((0.18, 0.055, 0.07, 0.05))
            play_ax = self.fig.add_axes((0.27, 0.055, 0.10, 0.05))
            next_ax = self.fig.add_axes((0.39, 0.055, 0.07, 0.05))
            slider_ax = self.fig.add_axes((0.54, 0.055, 0.37, 0.05))
            self.previous = Button(previous_ax, "Prev")
            self.play = Button(play_ax, "Play")
            self.next = Button(next_ax, "Next")
            self.slider = Slider(slider_ax, "tick", 0, max(1, len(frames) - 1), valinit=0, valstep=1)
            self.previous.on_clicked(lambda _event: self.set_frame(self.index - 1))
            self.next.on_clicked(lambda _event: self.set_frame(self.index + 1))
            self.play.on_clicked(lambda _event: self.toggle_play())
            self.slider.on_changed(self._slider_changed)
            self.fig.canvas.mpl_connect("button_press_event", self._click)
            self.fig.canvas.mpl_connect("key_press_event", self._key)
            self.timer = self.fig.canvas.new_timer(interval=round(1000 / fps))
            self.timer.add_callback(self._advance)
        self.render(0, 0.0)

    def _slider_changed(self, value: float) -> None:
        if not self._updating_slider:
            self.set_frame(int(value))

    def _sync_slider(self) -> None:
        if self.slider is not None and int(self.slider.val) != self.index:
            self._updating_slider = True
            try:
                self.slider.set_val(self.index)
            finally:
                self._updating_slider = False

    def set_frame(self, index: int) -> None:
        if self.playing:
            self.toggle_play()
        self.index = max(0, min(len(self.frames) - 1, index))
        self.subframe = 0
        self._sync_slider()
        self.render(self.index, 0.0)

    def toggle_play(self) -> None:
        if self.timer is None:
            return
        self.playing = not self.playing
        self.play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            if self.index == len(self.frames) - 1:
                self.index = 0
                self.subframe = 0
                self._sync_slider()
                self.render(0, 0.0)
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _advance(self) -> None:
        self.subframe += 1
        if self.subframe >= self.subframes:
            self.subframe = 0
            self.index += 1
            self._sync_slider()
        if self.index >= len(self.frames) - 1:
            self.index = len(self.frames) - 1
            self.subframe = 0
            self.render(self.index, 0.0)
            self.toggle_play()
        else:
            self.render(self.index, self.subframe / self.subframes)

    def _key(self, event) -> None:
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self.set_frame(self.index - 1)
        elif event.key == "right":
            self.set_frame(self.index + 1)
        elif event.key == "escape":
            self.selected_node = self.selected_task = None
            self._set_normalization()
            self.render(self.index, self.subframe / self.subframes)

    def _click(self, event) -> None:
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return
        def pixels(point):
            x, y = self.ax.transData.transform(point)
            return math.hypot(x - event.x, y - event.y)
        task_hit = min(self._task_hit, key=lambda key: pixels(self._task_hit[key]), default=None)
        if task_hit is not None and pixels(self._task_hit[task_hit]) < 11:
            self.selected_task = None if self.selected_task == task_hit else task_hit
        else:
            node_hit = min(self.positions, key=lambda key: pixels(self.positions[key]))
            if pixels(self.positions[node_hit]) >= 12:
                return
            self.selected_node = None if self.selected_node == node_hit else node_hit
        self._set_normalization()
        self.render(self.index, self.subframe / self.subframes)

    @staticmethod
    def _between(positions, source, target, depart, arrive, display_tick):
        fraction = max(0.0, min(1.0, (display_tick - depart) / (arrive - depart)))
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        return (x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0))

    def _field_values(self, frame: dict):
        nodes = frame["nodes"]
        task = frame["tasks"].get(self.selected_task) if self.selected_task else None
        belief = frame["beliefs"].get(self.selected_node, {}) if self.selected_node else None
        values = {}
        infeasible = set()
        unknown = set()
        for node_id, node in nodes.items():
            state = belief.get(node_id) if belief is not None else node
            if state is None:
                unknown.add(node_id)
                continue
            rate = state["cpu_rate"]
            capacity = state["memory_capacity"]
            work = state["known_remaining_work"] if belief is not None else state["remaining_work"]
            if task is not None:
                if task["memory_required"] > capacity:
                    infeasible.add(node_id)
                    continue
                if (task["current_node"] == node_id and task["status"] in ("queued", "running")
                    and (belief is None or self.selected_node == node_id)):
                    work -= task["remaining_work"]
                work += task["compute_work"]
            values[node_id] = work / rate
        return values, infeasible, unknown, task

    def _set_normalization(self) -> None:
        if self.selected_node is None and self.selected_task is None:
            self.norm = self.truth_norm
        else:
            samples = []
            for frame in self.frames:
                if self.selected_task is not None:
                    task = frame["tasks"].get(self.selected_task)
                    if task is None or task["status"] == "completed":
                        continue
                values, _, _, _ = self._field_values(frame)
                samples.extend(values.values())
            if samples:
                low, high = min(samples), max(samples)
                if high - low < 1e-9:
                    low, high = low - 0.5, high + 0.5
                self.norm = Normalize(low, high, clip=True)
            else:
                self.norm = self.truth_norm
        self._rendered_selection = None

    def render(self, index: int, alpha: float = 0.0, request_draw: bool = True) -> None:
        """Render one discrete snapshot, moving only in-flight markers by alpha."""
        self.index = index
        frame = self.frames[index]
        if self.selected_task is not None:
            task = frame["tasks"].get(self.selected_task)
            if task is None or task["status"] == "completed":
                self.selected_task = None
                self._set_normalization()
        selection = (self.selected_node, self.selected_task)
        if index != self._rendered_index or selection != self._rendered_selection:
            self._draw_discrete(frame)
            self._rendered_index = index
            self._rendered_selection = selection
        self._update_motion(frame, frame["tick"] + alpha)
        if request_draw:
            self.fig.canvas.draw_idle()

    def _draw_discrete(self, frame: dict) -> None:
        self.ax.clear()
        nodes = frame["nodes"]
        values, infeasible, unknown, task = self._field_values(frame)
        # Triangulate valid samples only; unknown and infeasible nodes never receive values.
        ids = list(values)
        if len(ids) >= 3:
            try:
                tri = mtri.Triangulation([self.positions[key][0] for key in ids],
                                         [self.positions[key][1] for key in ids])
                levels = [self.norm.vmin + (self.norm.vmax - self.norm.vmin) * k / 14
                          for k in range(15)]
                self.ax.tricontourf(tri, [values[key] for key in ids], levels=levels,
                                    cmap=self.cmap, norm=self.norm, alpha=0.68, extend="both")
            except (ValueError, RuntimeError):
                pass  # Tiny or collinear graphs still show discrete samples.
        for node_id, node in nodes.items():
            for peer in node["neighbors"]:
                if node_id < peer:
                    a, b = self.positions[node_id], self.positions[peer]
                    self.ax.plot((a[0], b[0]), (a[1], b[1]), color="0.45", lw=0.8, alpha=0.5, zorder=2)
        for node_id, pressure in values.items():
            x, y = self.positions[node_id]
            self.ax.scatter([x], [y], s=95, c=[self.cmap(self.norm(pressure))],
                            edgecolors="black", linewidths=0.8, zorder=4)
        for node_id in infeasible:
            x, y = self.positions[node_id]
            self.ax.scatter([x], [y], marker="X", c="crimson", s=95, zorder=4)
        for node_id in unknown:
            x, y = self.positions[node_id]
            self.ax.scatter([x], [y], marker="o", facecolors="none", edgecolors="0.4", s=95, zorder=4)
        for node_id, (x, y) in self.positions.items():
            self.ax.annotate(node_id, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7, zorder=5)
        if self.selected_node:
            x, y = self.positions[self.selected_node]
            self.ax.scatter([x], [y], s=230, facecolors="none", edgecolors="orange", lw=2, zorder=6)
        self._static_task_hit = {}
        at_node = {}
        for task_id, item in frame["tasks"].items():
            if item["status"] in ("queued", "running") and item["current_node"] in self.positions:
                node_id = item["current_node"]
                number = at_node.get(node_id, 0)
                at_node[node_id] = number + 1
                x, y = self.positions[node_id]
                angle = (number % 8) * math.tau / 8
                radius = 0.13 + 0.08 * (number // 8)
                self._static_task_hit[task_id] = (x + radius * math.cos(angle), y + radius * math.sin(angle))
        for task_id, (x, y) in self._static_task_hit.items():
            item = frame["tasks"][task_id]
            color = "gold" if item["status"] != "running" else "darkorange"
            self.ax.scatter([x], [y], c=color, s=48 if task_id == self.selected_task else 20,
                            edgecolors="black" if task_id == self.selected_task else "none", zorder=7)
        self._gossip_artist = self.ax.scatter([], [], c="deepskyblue", s=9, alpha=0.45, zorder=3)
        self._transit_artist = self.ax.scatter([], [], c="gold", s=20, zorder=7)
        self._selected_transit_artist = self.ax.scatter([], [], c="gold", s=48,
                                                        edgecolors="black", zorder=8)
        mode = "truth" if self.selected_node is None else f"{self.selected_node} belief"
        metric = "pressure" if task is None else f"projected pressure for {task['task_id']} (mem {task['memory_required']})"
        self.ax.set_title(f"Tick {frame['tick']} · {mode} · {metric}")
        self.ax.set_xlabel("CPU rate (work / tick)")
        self.ax.set_ylabel("Memory capacity")
        xs, ys = zip(*self.positions.values())
        self.ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        self.ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
        self.ax.grid(alpha=0.12)
        self.ax.text(0.01, 0.01, "Click node: truth/belief  ·  Click task: projected cost  ·  Esc: clear",
                     transform=self.ax.transAxes, fontsize=8, va="bottom",
                     bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        self.ax.text(0.99, 0.01,
                     f"purple low {self.norm.vmin:.1f}  ·  yellow high {self.norm.vmax:.1f} ticks",
                     transform=self.ax.transAxes, fontsize=8, va="bottom", ha="right",
                     bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    def _update_motion(self, frame: dict, display_tick: float) -> None:
        def offsets(points):
            return np.asarray(points).reshape((-1, 2))
        gossip = [self._between(self.positions, m["source"], m["target"],
                                m["depart_tick"], m["arrive_tick"], display_tick)
                  for m in frame["gossip_messages"]]
        transits = {t["task_id"]: self._between(self.positions, t["source"], t["target"],
                                                t["depart_tick"], t["arrive_tick"], display_tick)
                    for t in frame["task_transits"]}
        self._gossip_artist.set_offsets(offsets(gossip))
        self._transit_artist.set_offsets(offsets(list(transits.values())))
        selected = transits.get(self.selected_task)
        self._selected_transit_artist.set_offsets(offsets([selected] if selected else []))
        self._task_hit = {**self._static_task_hit, **transits}


class _StreamingPillowWriter(PillowWriter):
    """PillowWriter with bounded memory for long replays."""

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._output = open(self.outfile, "wb")
        self._first_frame = True

    def grab_frame(self, **savefig_kwargs):
        buffer = BytesIO()
        self.fig.savefig(buffer, format="rgba", dpi=self.dpi, **savefig_kwargs)
        image = Image.frombuffer("RGBA", self.frame_size, buffer.getbuffer(), "raw", "RGBA", 0, 1)
        image = image.convert("RGB").quantize(colors=256)
        if self._first_frame:
            header, _ = GifImagePlugin.getheader(image, info={"loop": 0})
            for block in header:
                self._output.write(block)
            self._first_frame = False
        # Each frame has its own palette and is written immediately. PillowWriter's
        # default finish() would otherwise keep every full-size frame in memory.
        GifImagePlugin._write_frame_data(
            self._output, image, (0, 0),
            {"duration": round(1000 / self.fps), "disposal": 2, "include_color_table": True})

    def finish(self):
        self._output.write(b";")
        self._output.close()


def save_gif(frames: list[dict], path, fps: int = DEFAULT_FPS) -> None:
    """Export the same replay renderer at visual subframe cadence."""
    viewer = ReplayViewer(frames, fps=fps, controls=False)
    steps = [(index, part / viewer.subframes)
             for index in range(len(frames) - 1) for part in range(viewer.subframes)]
    steps.append((len(frames) - 1, 0.0))
    animation = FuncAnimation(viewer.fig,
                              lambda step: viewer.render(*step, request_draw=False),
                              frames=steps, interval=1000 / fps, repeat=False, blit=False)
    try:
        animation.save(path, writer=_StreamingPillowWriter(fps=fps), dpi=80)
    finally:
        plt.close(viewer.fig)


def show(frames: list[dict], fps: int = DEFAULT_FPS) -> None:
    ReplayViewer(frames, fps=fps)
    plt.show()
