"""Reproducibility tests.

Verifies that same seed + same config → same game trajectory and agent output.

Tests:
    1. Game engine: two resets with same seed produce identical initial states.
    2. Game engine: full game trajectory is identical across two runs.
    3. SeedContext: isolated seeded block does not leak into global state.
    4. SeedContext: same seed in two contexts produces identical random output.
    5. apply_seed: global seed produces identical sequences across calls.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig, AgentConfig
from game.engine import GameEngine
from game.entities import GameAction, ActionType, Direction
from seed_utils import apply_seed, SeedContext, get_seed_state, restore_seed_state


def _run_game_trajectory(seed: int) -> list[tuple[int, int, int]]:
    """Run a short game and return (turn, player_x, player_y, player_hp) per turn."""
    config = GameConfig(
        grid_width=6, grid_height=6,
        num_enemies=2, num_health_packs=1,
        max_turns=10, seed=seed,
    )
    engine = GameEngine(config)
    state = engine.reset()
    trajectory = [(state.turn, state.player.x, state.player.y, state.player.hp)]

    actions = [
        GameAction(ActionType.MOVE, Direction.EAST),
        GameAction(ActionType.MOVE, Direction.SOUTH),
        GameAction(ActionType.WAIT),
        GameAction(ActionType.MOVE, Direction.EAST),
        GameAction(ActionType.MOVE, Direction.SOUTH),
    ]
    for action in actions:
        if state.game_over:
            break
        state = engine.step(action)
        trajectory.append((state.turn, state.player.x, state.player.y, state.player.hp))

    return trajectory


def test_same_seed_same_initial_state():
    """Two engine resets with the same seed must produce identical initial states."""
    config = GameConfig(seed=42)

    engine_a = GameEngine(config)
    state_a = engine_a.reset()

    engine_b = GameEngine(config)
    state_b = engine_b.reset()

    assert state_a.player.x == state_b.player.x
    assert state_a.player.y == state_b.player.y
    assert state_a.goal.x == state_b.goal.x
    assert state_a.goal.y == state_b.goal.y
    assert [(e.x, e.y) for e in state_a.enemies] == [(e.x, e.y) for e in state_b.enemies]
    assert [(h.x, h.y) for h in state_a.health_packs] == [(h.x, h.y) for h in state_b.health_packs]
    print("✓ test_same_seed_same_initial_state passed")


def test_same_seed_same_trajectory():
    """Same seed must produce byte-identical game trajectories."""
    traj_a = _run_game_trajectory(seed=7)
    traj_b = _run_game_trajectory(seed=7)
    assert traj_a == traj_b, f"Trajectories differ:\n  A: {traj_a}\n  B: {traj_b}"
    print("✓ test_same_seed_same_trajectory passed")


def test_different_seeds_different_trajectories():
    """Different seeds should (very likely) produce different trajectories."""
    traj_a = _run_game_trajectory(seed=1)
    traj_b = _run_game_trajectory(seed=999)
    # It's theoretically possible (but extremely unlikely) for two seeds to
    # produce the same 6x6 layout — so we just check they're not always equal.
    # If this flakes, increase grid size or turn count.
    assert traj_a != traj_b, "Different seeds produced identical trajectories — check RNG"
    print("✓ test_different_seeds_different_trajectories passed")


def test_seed_context_isolation():
    """SeedContext must not leak its seed into the surrounding global state."""
    apply_seed(0)
    before = [random.random() for _ in range(5)]

    apply_seed(0)
    with SeedContext(seed=99):
        _ = [random.random() for _ in range(10)]  # consume inside context
    after = [random.random() for _ in range(5)]

    assert before == after, "SeedContext leaked into global RNG state"
    print("✓ test_seed_context_isolation passed")


def test_seed_context_reproducibility():
    """Two SeedContext blocks with the same seed must produce identical output."""
    with SeedContext(seed=42):
        result_a = [random.random() for _ in range(20)]

    with SeedContext(seed=42):
        result_b = [random.random() for _ in range(20)]

    assert result_a == result_b, "Same SeedContext seed produced different output"
    print("✓ test_seed_context_reproducibility passed")


def test_get_restore_state():
    """get_seed_state / restore_seed_state must allow exact RNG replay."""
    apply_seed(123)
    snapshot = get_seed_state()
    sequence_a = [random.random() for _ in range(10)]

    restore_seed_state(snapshot)
    sequence_b = [random.random() for _ in range(10)]

    assert sequence_a == sequence_b, "Restored state produced different sequence"
    print("✓ test_get_restore_state passed")


def test_agent_config_seed_field():
    """AgentConfig must accept seed and deterministic_mode without error."""
    cfg = AgentConfig(seed=42, deterministic_mode=False)
    assert cfg.seed == 42
    assert cfg.deterministic_mode is False
    print("✓ test_agent_config_seed_field passed")


if __name__ == "__main__":
    test_same_seed_same_initial_state()
    test_same_seed_same_trajectory()
    test_different_seeds_different_trajectories()
    test_seed_context_isolation()
    test_seed_context_reproducibility()
    test_get_restore_state()
    test_agent_config_seed_field()
    print("\n✅ All reproducibility tests passed!")
