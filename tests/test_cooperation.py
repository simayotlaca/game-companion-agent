import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.cooperation import (
    PlayerCommand, PlanProposal, RoleAdvisor, CooperationController
)
from agent.companion import CompanionAgent


def test_player_commands():
    valid_cases = [
        ("focus heal", "HEAL", None),
        ("heal", "HEAL", None),
        ("rush", None, "rush goal"),
        ("rush goal", None, "rush goal"),
        ("hold", "WAIT", None),
        ("hold position", "WAIT", None),
        ("attack", None, "aggressive"),
        ("fight", None, "aggressive"),
        ("retreat", None, "cautious"),
        ("fall back", None, "cautious"),
        ("ignore enemies", None, "rush goal"),
    ]

    for raw, expected_override, expected_intent in valid_cases:
        cmd = PlayerCommand(raw)
        assert cmd.is_valid, f"'{raw}' should be valid"
        assert cmd.override_action == expected_override, f"'{raw}' override: {cmd.override_action}"
        assert cmd.intent_override == expected_intent, f"'{raw}' intent: {cmd.intent_override}"
        print(f"  ✓ '{raw}' → {cmd}")

    cmd = PlayerCommand("dance wildly")
    assert not cmd.is_valid
    print(f"  ✓ 'dance wildly' → invalid")

    cmd = PlayerCommand("")
    assert not cmd.is_valid
    print(f"  ✓ '' → invalid")

    print("✓ test_player_commands passed")


def test_plan_proposal():
    proposal = PlanProposal(
        objective="Eliminate enemy #0",
        steps=["Attack enemy", "Continue if alive", "Move to goal"],
        risk_level="medium",
        suggested_action="ATTACK EAST",
        companion_rationale="Enemy in range and HP is healthy.",
    )

    text = proposal.present()
    assert "Eliminate enemy #0" in text
    assert "ATTACK EAST" in text
    assert "medium" in text
    assert "Approve" in text
    print(f"  ✓ Proposal text ({len(text)} chars)")

    proposal.approve()
    assert proposal.player_approved
    print(f"  ✓ Approved")

    proposal2 = PlanProposal(
        objective="Test", steps=["Step 1"], risk_level="low",
        suggested_action="MOVE NORTH", companion_rationale="test",
    )
    proposal2.modify("rush goal instead")
    assert proposal2.player_approved
    assert proposal2.player_modification == "rush goal instead"
    print(f"  ✓ Modified: {proposal2.player_modification}")

    print("✓ test_plan_proposal passed")


def test_role_advisor():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    msg = RoleAdvisor.suggest_role(state, "ATTACK", "SOUTH")
    assert len(msg) > 0
    print(f"  ✓ ATTACK: \"{msg}\"")

    msg = RoleAdvisor.suggest_role(state, "HEAL")
    assert len(msg) > 0
    print(f"  ✓ HEAL: \"{msg}\"")

    msg = RoleAdvisor.suggest_role(state, "MOVE")
    assert len(msg) > 0
    print(f"  ✓ MOVE: \"{msg}\"")

    msg = RoleAdvisor.suggest_role(state, "WAIT")
    assert len(msg) > 0
    print(f"  ✓ WAIT: \"{msg}\"")

    print("✓ test_role_advisor passed")


def test_cooperation_controller():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    coop = CooperationController(auto_approve=True)

    proposal = coop.propose_plan(
        state=state,
        plan_objective="Navigate to goal",
        plan_steps=["Move east", "Avoid enemies", "Reach goal"],
        risk_level="low",
        suggested_action="MOVE EAST",
        rationale="No immediate threats.",
    )
    assert proposal.player_approved
    print(f"  ✓ Auto-approved proposal")

    coop.receive_player_input("rush goal")
    assert coop.current_command is not None
    assert coop.current_command.intent_override == "rush goal"
    print(f"  ✓ Command received: {coop.current_command}")

    context = coop.get_turn_context()
    assert "rush goal" in context.lower()
    print(f"  ✓ Turn context: \"{context}\"")

    coop.clear_turn_command()
    assert coop.current_command is None
    print(f"  ✓ Command cleared")

    msg = coop.get_role_message(state, "ATTACK", "EAST")
    assert len(msg) > 0
    print(f"  ✓ Role message: \"{msg}\"")

    print("✓ test_cooperation_controller passed")


def test_interactive_mode():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    responses = iter(["ok", "rush goal", "yes"])

    def mock_callback(plan_text: str) -> str:
        return next(responses)

    coop = CooperationController(auto_approve=False)
    coop.set_interactive(mock_callback)

    p1 = coop.propose_plan(state, "Plan A", ["step"], "low", "MOVE", "test")
    assert p1.player_approved
    assert coop.current_command is None
    print(f"  ✓ 'ok' → approved")

    p2 = coop.propose_plan(state, "Plan B", ["step"], "low", "MOVE", "test")
    assert p2.player_approved
    assert p2.player_modification == "rush goal"
    assert coop.current_command is not None
    assert coop.current_command.intent_override == "rush goal"
    print(f"  ✓ 'rush goal' → approved + command override")

    coop.clear_turn_command()

    p3 = coop.propose_plan(state, "Plan C", ["step"], "low", "MOVE", "test")
    assert p3.player_approved
    print(f"  ✓ 'yes' → approved")

    print("✓ test_interactive_mode passed")


def test_companion_with_commands():
    config = GameConfig(seed=42, max_turns=5)
    engine = GameEngine(config)
    renderer = GameRenderer(RenderConfig())
    agent_config = AgentConfig()

    agent = CompanionAgent(engine, renderer, agent_config, offline=True)
    agent.set_player_intent("balanced")

    state = engine.reset()

    state, log1 = agent.play_turn(state)
    assert log1.role_message != ""
    print(f"  ✓ Turn 1: {log1.chosen_action} | Role: \"{log1.role_message[:50]}...\"")

    agent.send_command("hold position")
    state, log2 = agent.play_turn(state)
    assert log2.player_command == "Hold position this turn"
    assert log2.chosen_action.action_type.name == "WAIT"
    print(f"  ✓ Turn 2: command='hold' → {log2.chosen_action}")

    state, log3 = agent.play_turn(state)
    assert log3.player_command == ""
    print(f"  ✓ Turn 3: command cleared → {log3.chosen_action}")

    agent.send_command("rush")
    state, log4 = agent.play_turn(state)
    assert log4.player_command != ""
    print(f"  ✓ Turn 4: command='rush' → {log4.chosen_action}")

    print("✓ test_companion_with_commands passed")


def test_companion_full_coop_game():
    config = GameConfig(seed=42, max_turns=8)
    engine = GameEngine(config)
    renderer = GameRenderer(RenderConfig())
    agent_config = AgentConfig()

    agent = CompanionAgent(engine, renderer, agent_config, offline=True)
    agent.set_player_intent("aggressive")

    final_state, logs = agent.play_game(max_turns=8, verbose=False)

    role_msgs = [log.role_message for log in logs if log.role_message]
    assert len(role_msgs) > 0, "Should have role messages"
    print(f"  ✓ Role messages: {len(role_msgs)}/{len(logs)} turns")

    plans = [log.plan for log in logs if log.plan]
    assert len(plans) > 0
    print(f"  ✓ Plans generated: {len(plans)}/{len(logs)} turns")

    assert agent.metrics.total_turns == len(logs)
    print(f"  ✓ Metrics: {agent.metrics.total_turns} turns, "
          f"{agent.metrics.fallback_rate:.1%} fallback")

    print("✓ test_companion_full_coop_game passed")


if __name__ == "__main__":
    test_player_commands()
    test_plan_proposal()
    test_role_advisor()
    test_cooperation_controller()
    test_interactive_mode()
    test_companion_with_commands()
    test_companion_full_coop_game()
    print("\n✅ All cooperation tests passed!")
