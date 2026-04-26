import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig, AgentConfig
from game.engine import GameEngine
from game.entities import GameAction, ActionType, Direction
from agent.reasoning import LLMReasoning, validate_schema, _normalize_output
from agent.action import ActionParser


def test_heuristic_reasoning():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    reasoning = LLMReasoning(AgentConfig(), offline=True)
    result = reasoning.decide("Test perception", turn=1, state=state)

    assert "action" in result
    assert "direction" in result
    assert "thinking" in result
    assert "companion_message" in result
    assert "plan" in result
    assert "risk_level" in result

    valid, err = validate_schema(result)
    assert valid, f"Schema validation failed: {err}"

    print(f"  Action: {result['action']} {result['direction']}")
    print(f"  Plan: {result['plan'][:2]}...")
    print(f"  Risk: {result['risk_level']}")
    print("✓ test_heuristic_reasoning passed")


def test_action_parser_valid():
    parser = ActionParser()

    test_cases = [
        ("MOVE NORTH", ActionType.MOVE, Direction.NORTH),
        ("ATTACK EAST", ActionType.ATTACK, Direction.EAST),
        ("HEAL", ActionType.HEAL, None),
        ("WAIT", ActionType.WAIT, None),
        ("move south", ActionType.MOVE, Direction.SOUTH),
    ]

    for action_str, expected_type, expected_dir in test_cases:
        parsed = parser._parse_action_string(action_str)
        assert parsed is not None, f"Failed to parse: {action_str}"
        assert parsed.action_type == expected_type
        assert parsed.direction == expected_dir
        print(f"  ✓ '{action_str}' → {parsed}")

    print("✓ test_action_parser_valid passed")


def test_action_parser_structured():
    parser = ActionParser()

    test_cases = [
        ({"action": "MOVE", "direction": "NORTH"}, ActionType.MOVE, Direction.NORTH),
        ({"action": "ATTACK", "direction": "EAST"}, ActionType.ATTACK, Direction.EAST),
        ({"action": "HEAL", "direction": None}, ActionType.HEAL, None),
        ({"action": "WAIT"}, ActionType.WAIT, None),
    ]

    for llm_output, expected_type, expected_dir in test_cases:
        parsed = parser._parse_structured(llm_output)
        assert parsed is not None, f"Failed to parse structured: {llm_output}"
        assert parsed.action_type == expected_type
        assert parsed.direction == expected_dir
        print(f"  ✓ {llm_output} → {parsed}")

    print("✓ test_action_parser_structured passed")


def test_action_parser_invalid():
    config = GameConfig(seed=42, num_enemies=0)
    engine = GameEngine(config)
    engine.reset()

    parser = ActionParser(fallback_enabled=True)

    result = parser.parse({"action": "DANCE WILDLY"}, engine)
    assert result is not None
    assert isinstance(result, GameAction)
    print(f"  ✓ Invalid input → fallback: {result}")
    print("✓ test_action_parser_invalid passed")


def test_schema_validation():
    valid_data = {
        "action": "MOVE", "direction": "NORTH",
        "thinking": "test", "companion_message": "test",
        "plan": ["step1"], "risk_level": "low",
    }
    ok, err = validate_schema(valid_data)
    assert ok, f"Should be valid: {err}"

    bad_data = {"action": "MOVE"}
    ok, err = validate_schema(bad_data)
    assert not ok
    print(f"  ✓ Missing field detected: {err}")

    bad_data2 = {
        "action": "DANCE", "direction": "NORTH",
        "thinking": "t", "companion_message": "t",
        "plan": [], "risk_level": "low",
    }
    ok, err = validate_schema(bad_data2)
    assert not ok
    print(f"  ✓ Invalid action detected: {err}")

    bad_data3 = {
        "action": "MOVE", "direction": None,
        "thinking": "t", "companion_message": "t",
        "plan": [], "risk_level": "low",
    }
    ok, err = validate_schema(bad_data3)
    assert not ok
    print(f"  ✓ MOVE without direction detected: {err}")

    print("✓ test_schema_validation passed")


def test_normalize_output():
    old = {"action": "MOVE NORTH", "thinking": "t", "companion_message": "m"}
    norm = _normalize_output(old)
    assert norm["action"] == "MOVE"
    assert norm["direction"] == "NORTH"
    assert "plan" in norm
    assert "risk_level" in norm
    print(f"  ✓ Old format normalized: {norm['action']} {norm['direction']}")
    print("✓ test_normalize_output passed")


def test_player_intent():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    reasoning = LLMReasoning(AgentConfig(), offline=True)

    result_normal = reasoning.decide("Test", turn=1, state=state)

    result_aggressive = reasoning.decide(
        "Test", turn=1, state=state, player_intent="aggressive"
    )

    ok1, _ = validate_schema(result_normal)
    ok2, _ = validate_schema(result_aggressive)
    assert ok1 and ok2

    print(f"  Normal:     {result_normal['action']} {result_normal['direction']}")
    print(f"  Aggressive: {result_aggressive['action']} {result_aggressive['direction']}")
    print("✓ test_player_intent passed")


def test_full_agent_turn():
    from agent.perception import VLMPerception
    from game.renderer import GameRenderer
    from config import RenderConfig

    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    renderer = GameRenderer(RenderConfig())
    perception = VLMPerception(AgentConfig(), offline=True)
    reasoning = LLMReasoning(AgentConfig(), offline=True)
    parser = ActionParser()

    screenshot = renderer.render(state)
    desc = perception.perceive(screenshot, state=state)

    decision = reasoning.decide(
        desc, turn=1, state=state,
        memory_context="Companion persona: balanced\nHP trend: STABLE",
        plan_context="Current plan: Navigate to goal",
        player_intent="cautious",
    )

    ok, err = validate_schema(decision)
    assert ok, f"Full turn schema failed: {err}"

    action = parser.parse(decision, engine)

    new_state = engine.step(action)

    assert new_state.turn == 1
    print(f"  ✓ Turn completed: {action} → {new_state.last_result[:80]}...")
    print(f"  ✓ Plan: {decision.get('plan', [])[:2]}")
    print(f"  ✓ Risk: {decision.get('risk_level', 'unknown')}")
    print("✓ test_full_agent_turn passed")


if __name__ == "__main__":
    test_heuristic_reasoning()
    test_action_parser_valid()
    test_action_parser_structured()
    test_action_parser_invalid()
    test_schema_validation()
    test_normalize_output()
    test_player_intent()
    test_full_agent_turn()
    print("\n✅ All reasoning tests passed!")
