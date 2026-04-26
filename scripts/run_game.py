import argparse
import os
import sys

from config import AppConfig, GameConfig, AgentConfig, RenderConfig, MODEL_REGISTRY
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.companion import CompanionAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Game Companion Agent")
    parser.add_argument("--turns", type=int, default=30, help="Max turns")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid width/height")
    parser.add_argument("--enemies", type=int, default=3, help="Number of enemies")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use-api", action="store_true", help="Use API endpoints")
    parser.add_argument("--4bit", dest="quantize", action="store_true",
                        help="Load models in 4-bit (lower VRAM)")
    parser.add_argument("--offline", action="store_true",
                        help="Run with rule-based agents (no GPU needed)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic mode (same seed → same output)")
    parser.add_argument("--save-screenshots", action="store_true",
                        help="Save rendered screenshots each turn")
    parser.add_argument("--vlm", type=str, default=None,
                        help=f"VLM model name or registry key {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--llm", type=str, default=None,
                        help=f"LLM model name or registry key {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--intent", type=str, default="",
                        help="Player intent: aggressive, cautious, rush goal, etc.")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: confirm companion plans each turn")
    return parser.parse_args()


def main():
    args = parse_args()

    game_config = GameConfig(
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        max_turns=args.turns,
        num_enemies=args.enemies,
        seed=args.seed,
    )

    agent_config = AgentConfig(
        use_api=args.use_api,
        load_in_4bit=args.quantize,
        seed=args.seed,
        deterministic_mode=args.deterministic,
    )
    if args.vlm:
        agent_config.vlm_model_name = MODEL_REGISTRY.get(args.vlm, args.vlm)
    if args.llm:
        agent_config.llm_model_name = MODEL_REGISTRY.get(args.llm, args.llm)

    if args.use_api:
        agent_config.vlm_api_url = os.getenv("VLM_API_URL", agent_config.vlm_api_url)
        agent_config.llm_api_url = os.getenv("LLM_API_URL", agent_config.llm_api_url)

    render_config = RenderConfig()

    engine = GameEngine(game_config)
    renderer = GameRenderer(render_config)
    agent = CompanionAgent(
        engine=engine,
        renderer=renderer,
        config=agent_config,
        offline=args.offline,
    )

    if args.intent:
        agent.set_player_intent(args.intent)

    if args.interactive:
        def interactive_callback(plan_text: str) -> str:
            print(f"\n{plan_text}")
            return input("  Your response (ok/command): ").strip()
        agent.set_interactive(interactive_callback)

    verbose = args.verbose and not args.quiet

    final_state, logs = agent.play_game(
        max_turns=args.turns,
        verbose=verbose,
    )

    if args.save_screenshots:
        os.makedirs("screenshots", exist_ok=True)
        state = engine.reset()
        for i, log in enumerate(logs):
            img = renderer.render(state)
            img.save(f"screenshots/turn_{i:03d}.png")
            if log.chosen_action:
                state = engine.step(log.chosen_action)

    return 0 if final_state.won else 1


if __name__ == "__main__":
    sys.exit(main())
