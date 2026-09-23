"""Precompute a deterministic run and open its Matplotlib replay."""
import argparse
from .scenario import default_scenario
from .simulator import Simulator

def main() -> None:
    parser = argparse.ArgumentParser(description="Basin distributed scheduling experiment")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gif", metavar="PATH", help="save the replay as a GIF before opening the viewer")
    parser.add_argument("--fps", type=int, default=20, help="visual playback and GIF frames per second")
    parser.add_argument("--gif-only", action="store_true", help="export without opening the viewer")
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.gif_only and not args.gif:
        parser.error("--gif-only requires --gif")
    frames = Simulator(default_scenario(args.seed)).run(args.ticks)
    from .viewer import save_gif, show
    if args.gif:
        save_gif(frames, args.gif, fps=args.fps)
    if not args.gif_only:
        show(frames, fps=args.fps)

if __name__ == "__main__":
    main()
