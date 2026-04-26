"""Visual replay demo for GridWorld Arena.

Runs the game offline (no GPU required) using rule-based AI,
then replays the recorded session in a Pygame window with glow
effects, animated entities, and a live AI companion HUD.

Usage:
    python run_visual.py [--turns N] [--seed N] [--speed N]

Options:
    --turns N    Max turns to simulate  (default: 30)
    --seed  N    Random seed            (default: 42)
    --speed N    Ms between turns       (default: 480)
    --headless   Render to PNG only, no window (useful for CI / screenshot)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer          # PIL renderer (used by agent)
from game.pygame_renderer import PygameRenderer  # Pygame renderer (visual)
from agent.companion import CompanionAgent


def run_game_capture(agent: CompanionAgent, max_turns: int):
    """Run a full game and return every intermediate state + all turn logs.

    Returns:
        states: list[GameState] — one entry per turn (including initial state)
        logs:   list[TurnLog | None] — None for initial, then one per turn
    """
    # Mirror the initialization that companion.play_game() does
    state = agent.engine.reset()
    agent.turn_logs.clear()
    agent.memory.reset()
    agent.strategy.reset()
    agent.metrics.reset()
    agent.coop.reset()

    states = [state]
    logs = [None]           # no log before first turn

    print(f"\n{'='*60}")
    print("  GridWorld Arena  —  AI Companion Agent")
    print(f"  Grid: {state.grid_width}x{state.grid_height} | "
          f"Enemies: {len(state.living_enemies)} | "
          f"Goal: ({state.goal.x},{state.goal.y})")
    print(f"{'='*60}\n")

    for _ in range(max_turns):
        if state.game_over:
            break
        state, log = agent.play_turn(state)
        states.append(state)
        logs.append(log)
        print(f"  Turn {log.turn:>3}  |  {log.chosen_action}  |  "
              f"HP {state.player.hp}/{state.player.max_hp}  |  "
              f"{log.companion_message[:60]}")

    print(f"\n{'='*60}")
    outcome = ("VICTORY" if state.won
               else "DEFEAT" if state.player.is_dead
               else "TURN LIMIT")
    print(f"  Outcome: {outcome}  |  Turns: {state.turn}  |  "
          f"Enemies defeated: "
          f"{sum(1 for e in state.enemies if not e.alive)}/{len(state.enemies)}")
    print(f"{'='*60}\n")

    return states, logs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual replay of GridWorld Arena AI companion")
    parser.add_argument("--turns",    type=int,  default=30,  help="Max turns")
    parser.add_argument("--seed",     type=int,  default=42,  help="Random seed")
    parser.add_argument("--speed",    type=int,  default=480, help="Ms between turns")
    parser.add_argument("--headless", action="store_true",
                        help="Skip window, save screenshot only")
    args = parser.parse_args()

    game_cfg = GameConfig(
        grid_width=8,
        grid_height=8,
        max_turns=args.turns,
        num_enemies=3,
        num_health_packs=2,
        seed=args.seed,
    )
    agent_cfg = AgentConfig()
    render_cfg = RenderConfig()

    engine   = GameEngine(game_cfg)
    pil_rend = GameRenderer(render_cfg)          # PIL renderer for VLM
    agent    = CompanionAgent(
        engine=engine,
        renderer=pil_rend,
        config=agent_cfg,
        offline=True,
    )

    print("Running offline AI simulation …")
    states, logs = run_game_capture(agent, max_turns=args.turns)

    if args.headless:
        # Render a single mid-game frame to PNG without opening a window
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    renderer = PygameRenderer(grid_w=game_cfg.grid_width,
                               grid_h=game_cfg.grid_height)

    if args.headless:
        mid = len(states) // 2
        renderer.render_frame(states[mid], logs[mid])
        import pygame
        pygame.image.save(renderer.screen, "gameplay_preview.png")
        print("Saved: gameplay_preview.png")
        renderer.close()
    else:
        print("Opening visual replay — use SPACE to advance, ESC to quit.\n")
        renderer.replay(states, logs,
                        turn_ms=args.speed,
                        screenshot_path="gameplay_preview.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
