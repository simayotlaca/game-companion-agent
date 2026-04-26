"""Generate an animated GIF showcasing the AI Companion in GridWorld Arena.

Run:
    python3 generate_gif.py

Output:
    gameplay.gif   — animated GIF ready for portfolio / README
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.companion import CompanionAgent


def parse_args():
    p = argparse.ArgumentParser(description="Generate gameplay GIF")
    p.add_argument("--output",    default="assets/gameplay.gif")
    p.add_argument("--seed",      type=int, default=200)
    p.add_argument("--turns",     type=int, default=30)
    p.add_argument("--fps",       type=int, default=2,
                   help="Frames per second (default 2)")
    p.add_argument("--hold-last", type=int, default=5,
                   help="Extra repeats of final frame (default 5)")
    p.add_argument("--cell-size", type=int, default=64)
    p.add_argument("--save-frames", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 62)
    print("  GridWorld Arena — GIF Generator")
    print(f"  seed={args.seed}  turns={args.turns}  fps={args.fps}")
    print("=" * 62)

    game_cfg   = GameConfig(
        grid_width=8, grid_height=8,
        max_turns=args.turns,
        num_enemies=3, num_health_packs=2,
        seed=args.seed,
    )
    render_cfg = RenderConfig(cell_size=args.cell_size)
    agent_cfg  = AgentConfig()

    engine   = GameEngine(game_cfg)
    renderer = GameRenderer(render_cfg)
    agent    = CompanionAgent(engine=engine, renderer=renderer,
                              config=agent_cfg, offline=True)

    if args.save_frames:
        os.makedirs("screenshots", exist_ok=True)

    # ── Reset and collect frames ──────────────────────────────────────────────
    frames: list[Image.Image] = []
    state = engine.reset()
    agent.memory.reset()
    agent.strategy.reset()
    agent.metrics.reset()
    agent.coop.reset()
    agent.turn_logs.clear()

    # Intro title card (1 frame, short hold)
    intro_hold = 1
    intro = renderer.make_intro_frame(state)
    frames.append(intro)
    if args.save_frames:
        intro.save("screenshots/turn_intro.png")
    print("  Frame INTRO — title card")

    # Turn 0: initial board
    f0 = renderer.render_composite(state, log=None, prev_state=None)
    frames.append(f0)
    if args.save_frames:
        f0.save("screenshots/turn_000.png")
    print("  Frame 000   — initial state")

    score = 0
    prev_state = state
    frame_actions = []  # track action type per frame for timing
    frame_actions.append("INIT")   # turn 0

    for t in range(args.turns):
        if state.game_over:
            break

        prev_enemies = len(prev_state.living_enemies)
        state, log = agent.play_turn(state)
        curr_enemies = len(state.living_enemies)

        kills = prev_enemies - curr_enemies
        if kills > 0:
            score += kills * 150
        if log and log.chosen_action and log.chosen_action.action_type.name == "HEAL":
            score += 25
        if state.won:
            score += 500

        frame = renderer.render_composite(state, log=log, prev_state=prev_state, score=score)
        frames.append(frame)

        act_name = (log.chosen_action.action_type.name if log and log.chosen_action else "NONE")
        frame_actions.append(act_name)

        if args.save_frames:
            frame.save(f"screenshots/turn_{t + 1:03d}.png")

        act = (
            log.chosen_action.action_type.name
            + (" " + log.chosen_action.direction.name
               if log.chosen_action and log.chosen_action.direction else "")
            if log.chosen_action else "—"
        )
        print(f"  Frame {t+1:03d}   — {act:24s}"
              f" HP {state.player.hp:3}/{state.player.max_hp} "
              f" E:{len(state.living_enemies)}")

        prev_state = state

    # Hold on final frame
    for _ in range(args.hold_last):
        frames.append(frames[-1])
        frame_actions.append("HOLD")

    # ── Save GIF — variable timing ────────────────────────────────────────────
    # Action frames (attack/heal/kill): slow and dramatic
    # Move frames: fast to keep things snappy
    ms_per_frame = max(100, 1000 // args.fps)

    durations = [600]  # intro title card
    for act in frame_actions:
        if act in ("ATTACK", "HEAL"):
            durations.append(700)
        elif act == "HOLD":
            durations.append(1200)
        else:
            durations.append(ms_per_frame)

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
    )

    outcome = ("Victory!" if state.won
               else "Defeat" if state.player.is_dead
               else "Turn limit")

    print()
    print(f"  Outcome  : {outcome}")
    print(f"  Frames   : {len(frames)}")
    print(f"  GIF size : {os.path.getsize(args.output) // 1024} KB")
    print(f"  Saved to : {args.output}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
