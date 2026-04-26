import time
from typing import Optional, Callable
from PIL import Image

from config import AgentConfig
from seed_utils import apply_seed
from game.engine import GameEngine, GameState
from game.renderer import GameRenderer
from agent.perception import VLMPerception
from agent.reasoning import LLMReasoning
from agent.action import ActionParser
from agent.memory import CompanionMemory
from agent.strategy import StrategyModule
from agent.metrics import MetricsLogger
from agent.cooperation import CooperationController, PlayerCommand
from game.entities import GameAction, ActionType


class TurnLog:

    def __init__(self, turn: int):
        self.turn = turn
        self.perception_output: str = ""
        self.reasoning_output: dict = {}
        self.chosen_action: Optional[GameAction] = None
        self.companion_message: str = ""
        self.action_was_fallback: bool = False
        self.game_result: str = ""
        self.plan: list[str] = []
        self.risk_level: str = ""
        self.persona: str = ""
        self.player_intent: str = ""

        self.coop_proposal: str = ""
        self.role_message: str = ""
        self.player_command: str = ""

    def __repr__(self) -> str:
        lines = [
            f"=== Turn {self.turn} ===",
            f"[Perception] VLM sees: \"{self.perception_output[:200]}\"",
            "",
            f"[Reasoning] LLM thinks: \"{self.reasoning_output.get('thinking', '')}\"",
        ]
        if self.plan:
            lines.append(f"[Plan] {' → '.join(self.plan[:3])}...")
        if self.risk_level:
            lines.append(f"[Risk] {self.risk_level}")
        if self.persona:
            lines.append(f"[Persona] {self.persona}")
        if self.player_intent:
            lines.append(f"[Player Intent] {self.player_intent}")
        if self.player_command:
            lines.append(f"[Player Command] {self.player_command}")
        lines.extend([
            "",
            f"[Action] {self.chosen_action}"
            + (" (fallback)" if self.action_was_fallback else ""),
            f"[Companion says] \"{self.companion_message}\"",
        ])
        if self.role_message:
            lines.append(f"[Role] \"{self.role_message}\"")
        lines.extend([
            "",
            f"[Result] {self.game_result}",
        ])
        return "\n".join(lines)


MAX_TURN_BUDGET_SECONDS: float = 60.0  # hard timeout per tick
MAX_PERCEPTION_LEN: int = 2000         # max chars from VLM to pass to LLM


class CompanionAgent:

    def __init__(self, engine: GameEngine, renderer: GameRenderer,
                 config: AgentConfig, offline: bool = False):
        self.engine = engine
        self.renderer = renderer
        self.perception = VLMPerception(config, offline=offline)
        self.reasoning = LLMReasoning(config, offline=offline)
        self.action_parser = ActionParser(
            fallback_enabled=config.fallback_to_heuristic
        )
        self.memory = CompanionMemory()
        self.strategy = StrategyModule()
        self.metrics = MetricsLogger()
        self.coop = CooperationController(auto_approve=True)
        self.config = config
        self.turn_logs: list[TurnLog] = []
        self._player_intent: str = ""
        self._decisions_this_tick: int = 0  # invariant: must be 1 after play_turn

        if config.seed is not None:
            apply_seed(config.seed, deterministic=config.deterministic_mode)

    def set_player_intent(self, intent: str):
        self._player_intent = intent

    def send_command(self, command: str):
        self.coop.receive_player_input(command)

    def set_interactive(self, callback: Callable[[str], str]):
        self.coop.set_interactive(callback)

    def play_turn(self, state: GameState) -> tuple[GameState, TurnLog]:
        # Contract: one decision per tick, game must not be over
        assert not state.game_over, "play_turn called on a finished game"
        self._decisions_this_tick = 0
        tick_start = time.monotonic()

        log = TurnLog(state.turn + 1)
        log.player_intent = self._player_intent
        log.persona = self.memory.persona

        self.metrics.start_turn(state.turn + 1)

        screenshot = self.renderer.render(state)

        self.metrics.record_vlm_start()
        perception_text = self.perception.perceive(screenshot, state=state)
        self.metrics.record_vlm_end()

        # Invariant: perception output must be non-empty string, truncated to budget
        assert isinstance(perception_text, str) and perception_text, "VLM returned empty perception"
        perception_text = perception_text[:MAX_PERCEPTION_LEN]
        log.perception_output = perception_text

        memory_context = self.memory.get_context(state)
        plan_context = self.strategy.get_plan_context(state)
        coop_context = self.coop.get_turn_context()

        effective_intent = self._player_intent
        player_cmd = self.coop.current_command
        if player_cmd and player_cmd.is_valid:
            log.player_command = player_cmd.description
            if player_cmd.intent_override:
                effective_intent = player_cmd.intent_override

        self.metrics.record_llm_start()

        if player_cmd and player_cmd.is_valid and player_cmd.override_action:
            reasoning_result = {
                "action": player_cmd.override_action,
                "direction": None,
                "thinking": f"Player commanded: {player_cmd.description}",
                "companion_message": f"Got it — {player_cmd.description.lower()}!",
                "plan": [player_cmd.description, "Resume normal strategy next turn"],
                "risk_level": "low",
            }
        else:
            full_context = effective_intent
            if coop_context:
                full_context = f"{effective_intent}\n{coop_context}" if effective_intent else coop_context

            reasoning_result = self.reasoning.decide(
                perception_text, state.turn + 1, state=state,
                memory_context=memory_context,
                plan_context=plan_context,
                player_intent=full_context,
            )
        self.metrics.record_llm_end()
        log.reasoning_output = reasoning_result
        log.plan = reasoning_result.get("plan", [])
        log.risk_level = reasoning_result.get("risk_level", "")

        plan = self.strategy.current_plan
        if plan:
            proposal = self.coop.propose_plan(
                state=state,
                plan_objective=plan.objective,
                plan_steps=plan.steps,
                risk_level=plan.risk_level,
                suggested_action=reasoning_result.get("action", "WAIT"),
                rationale=reasoning_result.get("thinking", ""),
            )
            log.coop_proposal = proposal.present()

        self.metrics.record_parse_start()
        action = self.action_parser.parse(reasoning_result, self.engine)
        self.metrics.record_parse_end()

        # Invariant: exactly one action decided per tick
        self._decisions_this_tick += 1
        assert self._decisions_this_tick == 1, (
            f"Multiple decisions in one tick: {self._decisions_this_tick}"
        )
        assert action is not None, "ActionParser returned None"

        # Timeout budget check
        elapsed = time.monotonic() - tick_start
        assert elapsed < MAX_TURN_BUDGET_SECONDS, (
            f"Turn exceeded budget: {elapsed:.1f}s > {MAX_TURN_BUDGET_SECONDS}s"
        )

        log.chosen_action = action
        log.companion_message = reasoning_result.get("companion_message", "")

        # A turn is a fallback only if the reasoning_result itself fails schema validation.
        # Re-parsing with _parse_structured incorrectly returns None for heuristic dicts
        # (already-parsed, not raw text), producing false-positive fallback counts.
        from agent.reasoning import validate_schema
        schema_valid, _ = validate_schema(reasoning_result)
        if not schema_valid:
            log.action_was_fallback = True

        action_name = action.action_type.name
        direction_name = action.direction.name if action.direction else None
        role_msg = self.coop.get_role_message(state, action_name, direction_name)
        log.role_message = role_msg

        new_state = self.engine.step(action)
        log.game_result = new_state.last_result

        self.memory.update(new_state, action.action_type)
        self.metrics.end_turn(
            action_type=action.action_type.name,
            was_fallback=log.action_was_fallback,
        )
        self.coop.clear_turn_command()

        self.turn_logs.append(log)
        return new_state, log

    def play_game(self, max_turns: Optional[int] = None,
                  verbose: bool = True) -> tuple[GameState, list[TurnLog]]:
        state = self.engine.reset()
        self.turn_logs.clear()
        self.memory.reset()
        self.strategy.reset()
        self.metrics.reset()
        self.coop.reset()
        turns = max_turns or self.engine.config.max_turns

        if verbose:
            print(f"\n{'='*60}")
            print(f"  GRIDWORLD ARENA — AI Companion Agent")
            print(f"  Grid: {state.grid_width}x{state.grid_height} | "
                  f"Enemies: {len(state.living_enemies)} | "
                  f"Goal: ({state.goal.x},{state.goal.y})")
            if self._player_intent:
                print(f"  Player Intent: {self._player_intent}")
            print(f"{'='*60}\n")

        for t in range(turns):
            if state.game_over:
                break

            state, log = self.play_turn(state)

            if verbose:
                print(log)
                print()

        self.metrics.record_game_result(
            won=state.won,
            total_turns=state.turn,
            final_hp=state.player.hp,
            enemies_defeated=sum(1 for e in state.enemies if not e.alive),
            total_enemies=len(state.enemies),
        )

        if verbose:
            self._print_summary(state)

        return state, self.turn_logs

    def _print_summary(self, state: GameState):
        print(f"\n{'='*60}")
        if state.won:
            print("  VICTORY! Agent reached the goal.")
        elif state.player.is_dead:
            print("  DEFEAT. Agent was eliminated.")
        else:
            print("  GAME OVER. Turn limit reached.")
        print(f"  Turns: {state.turn} | Final HP: {state.player.hp}/{state.player.max_hp}")
        print(f"  Enemies defeated: {sum(1 for e in state.enemies if not e.alive)}/{len(state.enemies)}")
        print(f"  Final persona: {self.memory.persona}")

        actions = [log.chosen_action.action_type.name for log in self.turn_logs if log.chosen_action]
        from collections import Counter
        dist = Counter(actions)
        print(f"  Actions: {dict(dist)}")
        fallbacks = sum(1 for log in self.turn_logs if log.action_was_fallback)
        if fallbacks:
            print(f"  Fallback actions: {fallbacks}/{len(self.turn_logs)}")

        print()
        print(self.metrics.summary())
        print(f"{'='*60}")
