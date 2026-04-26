import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig
from game.engine import GameEngine
from game.entities import GameAction, ActionType, Direction


def test_game_init():
    config = GameConfig(grid_width=8, grid_height=8, num_enemies=3,
                        num_health_packs=2, seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    assert state.player is not None
    assert state.player.hp == config.player_hp
    assert len(state.enemies) == config.num_enemies
    assert len(state.health_packs) == config.num_health_packs
    assert state.goal is not None
    assert state.turn == 0
    assert not state.game_over
    print("✓ test_game_init passed")


def test_player_movement():
    config = GameConfig(grid_width=8, grid_height=8, num_enemies=0,
                        num_health_packs=0, seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    initial_pos = (state.player.x, state.player.y)
    action = GameAction(ActionType.MOVE, Direction.EAST)
    new_state = engine.step(action)

    assert new_state.player.x == initial_pos[0] + 1 or "out of bounds" in new_state.last_result.lower()
    print("✓ test_player_movement passed")


def test_attack_mechanics():
    config = GameConfig(grid_width=8, grid_height=8, num_enemies=1,
                        num_health_packs=0, seed=42, player_attack_range=8)
    engine = GameEngine(config)
    state = engine.reset()

    enemy = state.enemies[0]
    initial_hp = enemy.hp

    dx = enemy.x - state.player.x
    dy = enemy.y - state.player.y
    if abs(dx) >= abs(dy):
        direction = Direction.EAST if dx > 0 else Direction.WEST
    else:
        direction = Direction.SOUTH if dy > 0 else Direction.NORTH

    action = GameAction(ActionType.ATTACK, direction)
    new_state = engine.step(action)

    attacked_enemy = new_state.enemies[0]
    assert attacked_enemy.hp < initial_hp or "no enemy in range" in new_state.last_result.lower()
    print("✓ test_attack_mechanics passed")


def test_legal_actions():
    config = GameConfig(grid_width=8, grid_height=8, num_enemies=1,
                        num_health_packs=1, seed=42)
    engine = GameEngine(config)
    engine.reset()

    legal = engine.get_legal_actions()
    assert len(legal) > 0
    assert any(a.action_type == ActionType.WAIT for a in legal)
    print("✓ test_legal_actions passed")


def test_game_over_on_goal():
    config = GameConfig(grid_width=4, grid_height=4, num_enemies=0,
                        num_health_packs=0, seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    for _ in range(20):
        if state.game_over:
            break
        dx = state.goal.x - state.player.x
        dy = state.goal.y - state.player.y
        if abs(dx) >= abs(dy):
            d = Direction.EAST if dx > 0 else Direction.WEST
        else:
            d = Direction.SOUTH if dy > 0 else Direction.NORTH
        state = engine.step(GameAction(ActionType.MOVE, d))

    assert state.won or state.game_over
    print("✓ test_game_over_on_goal passed")


def test_state_serialization():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    text = state.to_text()
    assert "Player" in text
    assert "Goal" in text
    assert "Turn" in text
    print("✓ test_state_serialization passed")


if __name__ == "__main__":
    test_game_init()
    test_player_movement()
    test_attack_mechanics()
    test_legal_actions()
    test_game_over_on_goal()
    test_state_serialization()
    print("\n✅ All engine tests passed!")
