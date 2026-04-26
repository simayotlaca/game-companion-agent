"""LLM reasoning module for action decision-making.

Provides multiple backends for converting perception text into structured
JSON action decisions: local Transformer, API-based, and heuristic (offline).
"""

import json
import logging
import re
from typing import Optional, Protocol, Union

from config import AgentConfig
from game.engine import GameState

logger = logging.getLogger(__name__)

ACTION_SCHEMA = {
    "type": "object",
    "required": ["action", "direction", "thinking", "companion_message",
                  "plan", "risk_level"],
    "properties": {
        "action": {"type": "string", "enum": ["MOVE", "ATTACK", "HEAL", "WAIT"]},
        "direction": {"type": ["string", "null"],
                      "enum": ["NORTH", "SOUTH", "EAST", "WEST", None]},
        "thinking": {"type": "string"},
        "companion_message": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
    }
}


def validate_schema(data: dict) -> tuple[bool, str]:
    """Validate LLM output against the action JSON schema.

    Args:
        data: Parsed JSON dictionary from LLM output.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    required = ACTION_SCHEMA["required"]
    for key in required:
        if key not in data:
            return False, f"Missing required field: {key}"

    action = data.get("action", "")
    valid_actions = ["MOVE", "ATTACK", "HEAL", "WAIT"]
    if action not in valid_actions:
        return False, f"Invalid action: {action}. Must be one of {valid_actions}"

    direction = data.get("direction")
    valid_dirs = ["NORTH", "SOUTH", "EAST", "WEST", None]
    if direction not in valid_dirs:
        return False, f"Invalid direction: {direction}"

    if action in ("MOVE", "ATTACK") and direction is None:
        return False, f"{action} requires a direction"

    plan = data.get("plan", [])
    if not isinstance(plan, list) or len(plan) > 5:
        return False, "plan must be a list with at most 5 items"

    return True, ""


REASONING_SYSTEM_PROMPT = """You are an AI companion in a tactical grid game. You play alongside a human player and must decide the best action based on the game state.

Available actions:
- MOVE: Move one tile (requires direction: NORTH/SOUTH/EAST/WEST)
- ATTACK: Attack an enemy in range (requires direction)
- HEAL: Use a nearby health pack (no direction needed)
- WAIT: Do nothing this turn (no direction needed)

Decision priorities:
1. If HP is critically low (<30%) and health pack is nearby, HEAL or MOVE toward it
2. If an enemy is in attack range and you're healthy enough, ATTACK
3. If enemies are too close but you're weak, MOVE away
4. Otherwise, MOVE toward the goal

You MUST respond with EXACTLY this JSON structure:
{
  "action": "MOVE",
  "direction": "NORTH",
  "thinking": "Brief tactical reasoning for this decision",
  "companion_message": "Short message to the player about what you're doing",
  "plan": ["Step 1", "Step 2", "Step 3"],
  "risk_level": "low"
}

Rules:
- "action" must be one of: MOVE, ATTACK, HEAL, WAIT
- "direction" must be NORTH, SOUTH, EAST, or WEST for MOVE/ATTACK. Use null for HEAL/WAIT.
- "plan" is your next 3-5 tactical steps (receding horizon plan)
- "risk_level" is your assessment: "low", "medium", or "high"
- Do NOT add any text outside the JSON object"""


class ReasoningModel(Protocol):
    """Protocol for reasoning backend implementations."""

    def reason(self, perception: str, turn: int,
               memory_context: str, plan_context: str,
               player_intent: str) -> dict: ...


class TransformerReasoning:
    """Local Transformer model backend for tactical reasoning.

    Loads an instruction-tuned LLM via HuggingFace and generates
    structured JSON action decisions from perception text.

    Args:
        config: Agent configuration with model name and inference settings.

    Raises:
        RuntimeError: If model loading fails (e.g. insufficient VRAM).
    """

    def __init__(self, config: AgentConfig) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = self._resolve_device(config.device)
        dtype_map = {"float16": torch.float16, "float32": torch.float32,
                     "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(config.llm_dtype, torch.float16)
        self.max_tokens = config.llm_max_new_tokens
        self.temperature = config.llm_temperature

        logger.info("Loading LLM %s on %s...", config.llm_model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
            load_kwargs = {"torch_dtype": self.dtype}

            if config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                )
            else:
                load_kwargs["device_map"] = self.device if self.device != "cpu" else None

            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name, **load_kwargs
            )
            if self.device == "cpu" and not config.load_in_4bit:
                self.model = self.model.to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error("Failed to load LLM: %s", e)
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    def reason(self, perception: str, turn: int,
               memory_context: str = "", plan_context: str = "",
               player_intent: str = "") -> dict:
        """Generate an action decision from perception text.

        Args:
            perception: Structured text description of the game state.
            turn: Current turn number.
            memory_context: Companion memory context string.
            plan_context: Previous plan context string.
            player_intent: Player's strategic preference.

        Returns:
            Validated action dictionary conforming to ACTION_SCHEMA.
        """
        prompt = self._build_prompt(perception, turn,
                                     memory_context, plan_context, player_intent)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_response(response)

    def _build_prompt(self, perception: str, turn: int,
                      memory_context: str = "", plan_context: str = "",
                      player_intent: str = "") -> str:
        """Assemble the full prompt with system instructions and context."""
        parts = [f"Turn {turn}. Current game state:\n{perception}"]
        if memory_context:
            parts.append(f"\nCompanion memory:\n{memory_context}")
        if plan_context:
            parts.append(f"\nPrevious plan:\n{plan_context}")
        if player_intent:
            parts.append(f"\nPlayer intent: {player_intent}")
        parts.append("\nDecide your action. Respond with JSON only.")
        user_msg = "\n".join(parts)
        return f"[INST] {REASONING_SYSTEM_PROMPT}\n\n{user_msg} [/INST]"

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse LLM text output into a validated action dictionary.

        Attempts JSON extraction first, falls back to regex pattern matching,
        and defaults to WAIT if all parsing fails.

        Args:
            text: Raw text output from the LLM.

        Returns:
            Action dictionary with all required schema fields.
        """
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                data = _normalize_output(data)
                valid, err = validate_schema(data)
                if valid:
                    return data
                logger.debug("Schema validation failed: %s", err)
            except json.JSONDecodeError as e:
                logger.debug("JSON parse failed: %s", e)

        action_match = re.search(
            r"(MOVE|ATTACK|HEAL|WAIT)\s*(NORTH|SOUTH|EAST|WEST)?",
            text, re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).upper()
            direction = action_match.group(2).upper() if action_match.group(2) else None
            logger.debug("Regex fallback: %s %s", action, direction)
            return {
                "action": action,
                "direction": direction,
                "thinking": text[:200],
                "companion_message": "Hmm, let me think about this...",
                "plan": [],
                "risk_level": "medium",
            }

        logger.warning("All parsing failed, defaulting to WAIT")
        return {
            "action": "WAIT",
            "direction": None,
            "thinking": text[:200],
            "companion_message": "I'm not sure what to do here.",
            "plan": [],
            "risk_level": "medium",
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to 'cuda' or 'cpu'."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device


def _normalize_output(data: dict) -> dict:
    """Normalize and fill missing fields in LLM output.

    Handles combined 'MOVE NORTH' action strings and ensures all
    required schema fields are present with sensible defaults.

    Args:
        data: Raw parsed JSON from LLM output.

    Returns:
        Normalized dictionary with all required fields.
    """
    action_str = data.get("action", "WAIT")
    if " " in action_str:
        parts = action_str.strip().split(None, 1)
        data["action"] = parts[0].upper()
        if len(parts) > 1 and "direction" not in data:
            data["direction"] = parts[1].upper()
    else:
        data["action"] = action_str.upper()

    if "direction" not in data:
        data["direction"] = None
    if "plan" not in data:
        data["plan"] = []
    if "risk_level" not in data:
        data["risk_level"] = "medium"
    if "thinking" not in data:
        data["thinking"] = ""
    if "companion_message" not in data:
        data["companion_message"] = ""

    return data


class APIReasoning:
    """API-based reasoning backend using OpenAI-compatible endpoints.

    Args:
        config: Agent configuration with API URL and inference settings.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.api_url = config.llm_api_url
        self.max_tokens = config.llm_max_new_tokens
        self.temperature = config.llm_temperature

    def reason(self, perception: str, turn: int,
               memory_context: str = "", plan_context: str = "",
               player_intent: str = "") -> dict:
        """Send perception to API and return the action decision.

        Args:
            perception: Structured text description of the game state.
            turn: Current turn number.
            memory_context: Companion memory context string.
            plan_context: Previous plan context string.
            player_intent: Player's strategic preference.

        Returns:
            Validated action dictionary.

        Raises:
            ConnectionError: If the API endpoint is unreachable.
        """
        import requests

        parts = [f"Turn {turn}. Current game state:\n{perception}"]
        if memory_context:
            parts.append(f"\nCompanion memory:\n{memory_context}")
        if plan_context:
            parts.append(f"\nPrevious plan:\n{plan_context}")
        if player_intent:
            parts.append(f"\nPlayer intent: {player_intent}")
        parts.append("\nDecide your action. Respond with JSON only.")
        user_msg = "\n".join(parts)

        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return TransformerReasoning._parse_response(text)
        except requests.ConnectionError as e:
            logger.error("LLM API unreachable at %s: %s", self.api_url, e)
            raise ConnectionError(f"LLM API unreachable: {self.api_url}") from e
        except (KeyError, IndexError) as e:
            logger.error("Unexpected LLM API response: %s", e)
            return {"action": "WAIT", "direction": None,
                    "thinking": "API response error", "companion_message": "Waiting...",
                    "plan": [], "risk_level": "medium"}


class HeuristicReasoning:
    """Offline rule-based reasoning backend (no model needed).

    Makes tactical decisions using BFS pathfinding + hand-coded priority
    rules based on HP ratios, enemy proximity, and player intent. Used for
    offline demos and testing.

    Decision priority (in order):
        1. Emergency heal
        2. Endgame sprint (goal <= 3 tiles away)
        3. Rush intent fast-path
        4. Focus-fire (with multi-enemy counter-damage check)
        5. Retreat if outnumbered or HP too low
        6. Close on path-blocking enemy
        7. BFS move toward goal (routing around enemies)
    """

    def __init__(self) -> None:
        self._last_state: Optional[GameState] = None

    def set_state(self, state: GameState) -> None:
        """Inject the current game state for heuristic reasoning."""
        self._last_state = state

    def reason(self, perception: str, turn: int,
               memory_context: str = "", plan_context: str = "",
               player_intent: str = "") -> dict:
        """Generate an action decision using heuristic + BFS rules.

        Args:
            perception: Ignored (state is used directly).
            turn: Current turn number.
            memory_context: Ignored in heuristic mode.
            plan_context: Ignored in heuristic mode.
            player_intent: Player intent string (affects engagement thresholds).

        Returns:
            Action dictionary conforming to ACTION_SCHEMA.
        """
        if self._last_state is None:
            return {"action": "WAIT", "direction": None,
                    "thinking": "No state", "companion_message": "Waiting...",
                    "plan": [], "risk_level": "low"}

        state = self._last_state
        p = state.player
        enemies = state.living_enemies
        health_packs = state.available_health_packs

        hp_ratio = p.hp / p.max_hp
        in_range_enemies = [e for e in enemies if p.distance_to(e) <= p.attack_range]
        # All enemies within melee+1 that will likely counter-attack if we stay
        threatening = [e for e in enemies if p.distance_to(e) <= 2]
        expected_counter_dmg = len(threatening) * 15  # enemy.attack_power

        nearby_enemies = [e for e in enemies if p.distance_to(e) <= p.attack_range + 2]
        risk = ("high" if (hp_ratio < 0.3 and nearby_enemies) else
                "medium" if (hp_ratio < 0.5 or len(nearby_enemies) >= 2) else "low")

        intent_aggressive = player_intent and "aggressive" in player_intent.lower()
        intent_cautious = player_intent and ("cautious" in player_intent.lower() or
                                              "stealth" in player_intent.lower() or
                                              "safe" in player_intent.lower())
        intent_rush = player_intent and ("rush" in player_intent.lower() or
                                         "goal" in player_intent.lower() or
                                         "speed" in player_intent.lower())

        goal_dist = p.distance_to(state.goal)

        # BFS path toward goal routing AROUND current enemy positions.
        # Falls back to greedy direction if all paths are blocked.
        bfs_dir = self._plan_move_to_goal(state)

        # ── 1. EMERGENCY HEAL ──────────────────────────────────────────────────
        heal_threshold = 0.25 if intent_aggressive else 0.35
        if hp_ratio < heal_threshold and health_packs:
            closest_hp = min(health_packs, key=lambda h: p.distance_to(h))
            if p.distance_to(closest_hp) <= 1:
                return {
                    "action": "HEAL", "direction": None,
                    "thinking": f"HP critically low ({p.hp}/{p.max_hp}). Health pack adjacent — healing now.",
                    "companion_message": f"Need to heal — only {p.hp} HP left!",
                    "plan": ["Heal now", "Reassess threats", "Resume objective"],
                    "risk_level": risk,
                }
            hp_dist = p.distance_to(closest_hp)
            if hp_dist <= 3 or not enemies:
                hp_dir = self._bfs_toward(state,
                                          blocked=frozenset(),
                                          target=(closest_hp.x, closest_hp.y))
                if hp_dir is None:
                    hp_dir = self._direction_toward(p.x, p.y, closest_hp.x, closest_hp.y)
                return {
                    "action": "MOVE", "direction": hp_dir,
                    "thinking": f"HP low ({p.hp}). Routing to health pack at ({closest_hp.x},{closest_hp.y}).",
                    "companion_message": "Low HP! Routing to the health pack.",
                    "plan": [f"Move to health pack ({closest_hp.x},{closest_hp.y})",
                             "Heal", "Reassess", "Push toward goal"],
                    "risk_level": risk,
                }

        # ── 2. ENDGAME SPRINT — goal is so close, just go ─────────────────────
        # When within 3 tiles, don't get distracted by combat — finish the run.
        if goal_dist <= 3 and hp_ratio > 0.2:
            sprint_dir = bfs_dir or self._direction_toward(p.x, p.y,
                                                            state.goal.x, state.goal.y)
            return {
                "action": "MOVE", "direction": sprint_dir,
                "thinking": (f"Goal is only {goal_dist} tile(s) away! "
                             f"Sprinting for the win via BFS route."),
                "companion_message": f"Almost there — {goal_dist} tiles to go!",
                "plan": [f"Sprint to goal ({state.goal.x},{state.goal.y})",
                         "Ignore distractions",
                         "Reach objective NOW"],
                "risk_level": risk,
            }

        # ── 3. RUSH INTENT — skip all combat, BFS beeline to goal ─────────────
        if intent_rush and hp_ratio > 0.3:
            rush_dir = bfs_dir or self._direction_toward(p.x, p.y,
                                                          state.goal.x, state.goal.y)
            return {
                "action": "MOVE", "direction": rush_dir,
                "thinking": f"Rush mode. Taking BFS route to goal, distance {goal_dist}.",
                "companion_message": "Rush mode — going straight for the objective!",
                "plan": [f"Sprint toward goal (distance: {goal_dist})",
                         "Avoid combat unless forced",
                         f"Reach ({state.goal.x},{state.goal.y})"],
                "risk_level": risk,
            }

        # ── 4. FOCUS-FIRE ATTACK with counter-damage assessment ────────────────
        # Before engaging, check if taking return fire from all threatening enemies
        # would be survivable. If not, retreat instead.
        engage_threshold = 0.2 if intent_aggressive else 0.25
        if in_range_enemies and hp_ratio > engage_threshold:
            hp_after_counter = p.hp - expected_counter_dmg
            # Fight if: counter damage is survivable, OR we're healthy enough,
            # OR the target is almost dead (finish the kill!)
            target = min(in_range_enemies, key=lambda e: e.hp)
            almost_dead = target.hp <= p.attack_power  # one-shot kill
            fight = (hp_after_counter > 0) or (hp_ratio > 0.6) or almost_dead
            if fight:
                direction = self._direction_toward(p.x, p.y, target.x, target.y)
                hp_pct = int(target.hp / target.max_hp * 100)
                return {
                    "action": "ATTACK", "direction": direction,
                    "thinking": (f"Focus-fire enemy #{target.entity_id} ({hp_pct}% HP). "
                                 f"Counter-damage estimate: {expected_counter_dmg} "
                                 f"({'survivable' if hp_after_counter > 0 else 'risky — one-shot kill'})."),
                    "companion_message": f"Focusing fire on #{target.entity_id} — {hp_pct}% HP!",
                    "plan": [f"Attack #{target.entity_id} (HP: {target.hp}/{target.max_hp})",
                             "Finish off if still alive",
                             "Sweep remaining threats",
                             f"Push to goal ({state.goal.x},{state.goal.y})"],
                    "risk_level": risk,
                }

        # ── 5. RETREAT — outnumbered, low HP, or counter-damage too high ──────
        # Use BFS retreat: move toward goal direction to combine flee + progress.
        outnumbered = len(threatening) >= 2 and hp_ratio < 0.5
        too_risky = in_range_enemies and (hp_ratio <= engage_threshold or outnumbered)
        if too_risky:
            # Try to flee toward goal (double benefit: retreat + progress)
            flee_dir = bfs_dir
            if flee_dir is None:
                flee_dir = self._direction_away(p.x, p.y,
                                                in_range_enemies[0].x,
                                                in_range_enemies[0].y)
            return {
                "action": "MOVE", "direction": flee_dir,
                "thinking": (f"HP={p.hp}, {len(threatening)} enemy/enemies threatening. "
                             f"Retreating via BFS path."),
                "companion_message": "Too risky — falling back toward safety!",
                "plan": ["Retreat from threats", "Find health pack", "Heal", "Re-engage"],
                "risk_level": risk,
            }

        # ── 6. CLOSE on PATH-BLOCKING enemies one step outside attack range ────
        # BFS already routes around enemies, but if an enemy is 1 step outside
        # our range AND on the BFS path, it's worth closing to attack next turn.
        kite_threshold = 0.3 if intent_cautious else 0.25
        if not intent_rush and hp_ratio > kite_threshold and enemies:
            half_goal_dist = max(1, goal_dist // 2)
            path_blockers = [
                e for e in enemies
                if p.attack_range < p.distance_to(e) <= p.attack_range + 1
                and p.distance_to(e) <= half_goal_dist + p.attack_range
                and self._direction_toward(p.x, p.y, e.x, e.y) == bfs_dir
            ]
            if path_blockers:
                target = min(path_blockers, key=lambda e: e.hp)
                move_dir = self._direction_toward(p.x, p.y, target.x, target.y)
                return {
                    "action": "MOVE", "direction": move_dir,
                    "thinking": (f"Enemy #{target.entity_id} is 1 step outside range on BFS path. "
                                 f"Closing to attack next turn."),
                    "companion_message": f"Path blocker ahead — closing to range on #{target.entity_id}!",
                    "plan": [f"Close on #{target.entity_id} (BFS path blocker)",
                             f"Attack #{target.entity_id}",
                             "Continue to goal"],
                    "risk_level": risk,
                }

        # ── 7. BFS MOVE TOWARD GOAL ────────────────────────────────────────────
        # BFS path routes around all current enemy positions automatically.
        move_dir = bfs_dir or self._direction_toward(p.x, p.y, state.goal.x, state.goal.y)
        return {
            "action": "MOVE", "direction": move_dir,
            "thinking": (f"No actionable threats. BFS route to goal at "
                         f"({state.goal.x},{state.goal.y}), distance {goal_dist}."),
            "companion_message": "Path clear — taking optimal route to objective.",
            "plan": [f"BFS move toward goal (distance: {goal_dist})",
                     "Engage threats only if in range",
                     "Pick up health packs on the way",
                     f"Reach goal at ({state.goal.x},{state.goal.y})"],
            "risk_level": risk,
        }

    # ── Pathfinding helpers ────────────────────────────────────────────────────

    @staticmethod
    def _bfs_toward(state: "GameState", blocked: frozenset,
                    target: tuple = None) -> Optional[str]:
        """BFS from player position; returns the first-step direction.

        Args:
            state: Current game state (supplies grid size and player pos).
            blocked: Set of (x,y) tiles to treat as impassable.
            target: (x,y) destination; defaults to state.goal.

        Returns:
            Cardinal direction string, or None if no path exists.
        """
        from collections import deque
        p = state.player
        tx, ty = target if target else (state.goal.x, state.goal.y)

        if (p.x, p.y) == (tx, ty):
            return None  # already there

        queue: deque = deque([(p.x, p.y, None)])
        visited: set = {(p.x, p.y)}

        DIRS = [(0, -1, "NORTH"), (0, 1, "SOUTH"), (1, 0, "EAST"), (-1, 0, "WEST")]

        while queue:
            x, y, first_dir = queue.popleft()
            for dx, dy, dname in DIRS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or (nx, ny) in blocked:
                    continue
                if not (0 <= nx < state.grid_width and 0 <= ny < state.grid_height):
                    continue
                fd = first_dir if first_dir is not None else dname
                if nx == tx and ny == ty:
                    return fd
                visited.add((nx, ny))
                queue.append((nx, ny, fd))

        return None  # no path

    @staticmethod
    def _plan_move_to_goal(state: "GameState") -> Optional[str]:
        """Return the BFS-optimal first step toward goal, routing around enemies.

        First tries with all living enemy positions blocked. If no path exists
        (fully surrounded), retries with only melee-adjacent (distance<=1) enemies
        blocked, then with no obstacles.

        Args:
            state: Current game state.

        Returns:
            Cardinal direction string, or None if completely stuck.
        """
        p = state.player
        all_enemy_pos = frozenset((e.x, e.y) for e in state.living_enemies)
        melee_only = frozenset((e.x, e.y) for e in state.living_enemies
                               if p.distance_to(e) <= 1)

        # Attempt 1: avoid all enemies
        d = HeuristicReasoning._bfs_toward(state, blocked=all_enemy_pos)
        if d:
            return d

        # Attempt 2: only avoid adjacent enemies (push through range-2 danger)
        d = HeuristicReasoning._bfs_toward(state, blocked=melee_only)
        if d:
            return d

        # Attempt 3: no blocking — just find the shortest path
        d = HeuristicReasoning._bfs_toward(state, blocked=frozenset())
        if d:
            return d

        # Total fallback: greedy direction
        return HeuristicReasoning._direction_toward(p.x, p.y, state.goal.x, state.goal.y)

    @staticmethod
    def _direction_toward(fx: int, fy: int, tx: int, ty: int) -> str:
        """Get cardinal direction from (fx,fy) toward (tx,ty)."""
        dx, dy = tx - fx, ty - fy
        if abs(dx) >= abs(dy):
            return "EAST" if dx > 0 else "WEST"
        return "SOUTH" if dy > 0 else "NORTH"

    @staticmethod
    def _direction_away(fx: int, fy: int, tx: int, ty: int) -> str:
        """Get cardinal direction from (fx,fy) away from (tx,ty)."""
        dx, dy = tx - fx, ty - fy
        if abs(dx) >= abs(dy):
            return "WEST" if dx > 0 else "EAST"
        return "NORTH" if dy > 0 else "SOUTH"


class LLMReasoning:
    """Unified reasoning interface that dispatches to the appropriate backend.

    Args:
        config: Agent configuration with model and API settings.
        offline: If True, use heuristic backend (no GPU/API needed).
    """

    def __init__(self, config: AgentConfig, offline: bool = False) -> None:
        if offline:
            self._backend: Union[HeuristicReasoning, APIReasoning, TransformerReasoning] = HeuristicReasoning()
        elif config.use_api:
            self._backend = APIReasoning(config)
        else:
            self._backend = TransformerReasoning(config)
        self._is_heuristic = offline

    def decide(self, perception: str, turn: int,
               state: Optional[GameState] = None,
               memory_context: str = "", plan_context: str = "",
               player_intent: str = "") -> dict:
        """Make an action decision based on perception and context.

        Args:
            perception: Structured text description of the game state.
            turn: Current turn number.
            state: Optional GameState for heuristic backend.
            memory_context: Companion memory context string.
            plan_context: Previous plan context string.
            player_intent: Player's strategic preference.

        Returns:
            Action dictionary conforming to ACTION_SCHEMA.
        """
        if self._is_heuristic and state is not None:
            self._backend.set_state(state)
        return self._backend.reason(perception, turn,
                                     memory_context=memory_context,
                                     plan_context=plan_context,
                                     player_intent=player_intent)
