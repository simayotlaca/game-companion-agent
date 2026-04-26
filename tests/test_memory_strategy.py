import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig
from game.engine import GameEngine
from game.entities import ActionType
from agent.memory import CompanionMemory
from agent.strategy import StrategyModule
from agent.metrics import MetricsLogger


def test_memory_persona_adaptation():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    memory = CompanionMemory(window_size=5)

    for _ in range(5):
        state.player.hp -= 15
        memory.update(state, ActionType.MOVE)

    assert memory.persona == "cautious", f"Expected cautious, got {memory.persona}"
    print(f"  ✓ Low HP → persona: {memory.persona}")

    memory.reset()
    state.player.hp = state.player.max_hp
    for _ in range(5):
        memory.update(state, ActionType.ATTACK)

    assert memory.persona == "aggressive", f"Expected aggressive, got {memory.persona}"
    print(f"  ✓ High HP + attacks → persona: {memory.persona}")
    print("✓ test_memory_persona_adaptation passed")


def test_memory_context_generation():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    memory = CompanionMemory()
    memory.update(state, ActionType.MOVE)
    memory.update(state, ActionType.ATTACK)
    memory.update(state, ActionType.MOVE)

    context = memory.get_context(state)
    assert "Companion persona:" in context
    assert "Enemies killed" in context
    assert "POLICY:" in context
    print(f"  ✓ Context generated ({len(context)} chars)")
    print("✓ test_memory_context_generation passed")


def test_strategy_plan_generation():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    strategy = StrategyModule()
    plan = strategy.generate_plan(state)

    assert plan.objective != ""
    assert len(plan.steps) >= 3
    assert plan.risk_level in ("low", "medium", "high")
    assert plan.next_best != ""

    print(f"  ✓ Objective: {plan.objective}")
    print(f"  ✓ Steps: {len(plan.steps)}")
    print(f"  ✓ Risk: {plan.risk_level}")
    print(f"  ✓ Fallback: {plan.next_best[:50]}...")
    print("✓ test_strategy_plan_generation passed")


def test_strategy_plan_context():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    strategy = StrategyModule()
    context = strategy.get_plan_context(state)

    assert "Current plan:" in context
    assert "Risk level:" in context
    assert "Steps:" in context
    assert "Fallback:" in context
    print(f"  ✓ Plan context generated ({len(context)} chars)")
    print("✓ test_strategy_plan_context passed")


def test_metrics_logging():
    metrics = MetricsLogger()

    for i in range(3):
        metrics.start_turn(i + 1)
        metrics.record_vlm_start()
        metrics.record_vlm_end()
        metrics.record_llm_start()
        metrics.record_llm_end()
        metrics.record_parse_start()
        metrics.record_parse_end()
        metrics.end_turn(
            action_type="MOVE" if i < 2 else "ATTACK",
            was_fallback=(i == 1),
        )

    assert metrics.total_turns == 3
    assert metrics.fallback_count == 1
    assert abs(metrics.fallback_rate - 1/3) < 0.01
    assert metrics.action_distribution == {"MOVE": 2, "ATTACK": 1}
    print(f"  ✓ Turns: {metrics.total_turns}")
    print(f"  ✓ Fallback rate: {metrics.fallback_rate:.1%}")
    print(f"  ✓ Distribution: {metrics.action_distribution}")

    metrics.record_game_result(
        won=True, total_turns=3, final_hp=80,
        enemies_defeated=1, total_enemies=3,
    )
    assert metrics.win_rate == 1.0

    summary = metrics.summary()
    assert "Performance Metrics" in summary
    assert "Fallback actions" in summary
    print(f"  ✓ Summary generated ({len(summary)} chars)")
    print("✓ test_metrics_logging passed")


def test_full_companion_with_features():
    from agent.companion import CompanionAgent
    from game.renderer import GameRenderer
    from config import AgentConfig, RenderConfig

    config = GameConfig(seed=42, max_turns=10)
    engine = GameEngine(config)
    renderer = GameRenderer(RenderConfig())
    agent_config = AgentConfig()

    agent = CompanionAgent(engine, renderer, agent_config, offline=True)
    agent.set_player_intent("aggressive")

    final_state, logs = agent.play_game(max_turns=10, verbose=False)

    assert len(logs) > 0
    assert all(log.plan is not None for log in logs)
    assert all(log.risk_level != "" for log in logs)
    assert agent.memory.persona in ("aggressive", "cautious", "balanced")
    assert agent.metrics.total_turns == len(logs)
    assert agent.metrics.avg_total_latency_ms >= 0

    print(f"  ✓ Played {len(logs)} turns")
    print(f"  ✓ Final persona: {agent.memory.persona}")
    print(f"  ✓ Win: {final_state.won}")
    print(f"  ✓ Fallback rate: {agent.metrics.fallback_rate:.1%}")
    print(f"  ✓ Avg latency: {agent.metrics.avg_total_latency_ms:.1f}ms")
    print("✓ test_full_companion_with_features passed")


if __name__ == "__main__":
    test_memory_persona_adaptation()
    test_memory_context_generation()
    test_strategy_plan_generation()
    test_strategy_plan_context()
    test_metrics_logging()
    test_full_companion_with_features()
    print("\n✅ All memory/strategy/metrics tests passed!")
