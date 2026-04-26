import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.companion import CompanionAgent


def main():
    print("=" * 60)
    print("  Game Companion Agent — Offline Demo")
    print("  (Rule-based perception & reasoning, no GPU needed)")
    print("=" * 60)

    game_config = GameConfig(
        grid_width=8,
        grid_height=8,
        max_turns=30,
        num_enemies=3,
        num_health_packs=2,
        seed=42,
    )

    agent_config = AgentConfig()
    render_config = RenderConfig()

    engine = GameEngine(game_config)
    renderer = GameRenderer(render_config)
    agent = CompanionAgent(
        engine=engine,
        renderer=renderer,
        config=agent_config,
        offline=True,
    )

    final_state, logs = agent.play_game(max_turns=30, verbose=True)

    try:
        final_img = renderer.render(final_state)
        final_img.save("final_state.png")
        print(f"\nFinal game state saved to: final_state.png")
    except Exception as e:
        print(f"\n(Could not save screenshot: {e})")

    print(f"\nTotal turns played: {len(logs)}")
    print(f"Outcome: {'Victory!' if final_state.won else 'Defeat' if final_state.player.is_dead else 'Turn limit'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
