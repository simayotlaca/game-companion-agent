"""Action parsing and validation module.

Converts LLM reasoning output into valid GameAction objects, with
multi-stage fallback: structured parse -> string parse -> closest legal -> heuristic.
"""

import logging
import re
from typing import Optional

from game.entities import GameAction, ActionType, Direction
from game.engine import GameEngine

logger = logging.getLogger(__name__)


class ActionParser:
    """Parses LLM output into legal game actions with fallback chain.

    The parsing pipeline:
    1. Try structured JSON fields (action + direction)
    2. Try regex on combined action string
    3. Find closest legal action of same type
    4. Fall back to heuristic priority ordering

    Args:
        fallback_enabled: Whether to use heuristic fallback as last resort.
    """

    def __init__(self, fallback_enabled: bool = True) -> None:
        self.fallback_enabled = fallback_enabled
        self._parse_pattern = re.compile(
            r"(MOVE|ATTACK|HEAL|WAIT)\s*(NORTH|SOUTH|EAST|WEST|UP|DOWN|LEFT|RIGHT|N|S|E|W)?",
            re.IGNORECASE
        )

    def parse(self, llm_output: dict, engine: GameEngine) -> GameAction:
        """Parse LLM output into a legal GameAction.

        Args:
            llm_output: Action dictionary from the reasoning module.
            engine: Game engine for legal action enumeration.

        Returns:
            A valid GameAction (falls back to WAIT if all parsing fails).
        """
        legal_actions = engine.get_legal_actions()

        parsed = self._parse_structured(llm_output)

        if parsed is None:
            action_str = llm_output.get("action", "WAIT")
            if isinstance(action_str, str):
                parsed = self._parse_action_string(action_str)

        if parsed and self._is_legal(parsed, legal_actions):
            return parsed

        if parsed:
            closest = self._find_closest_legal(parsed, legal_actions)
            if closest:
                logger.debug("Action corrected: %s -> %s", parsed, closest)
                return closest

        if self.fallback_enabled and legal_actions:
            fallback = self._heuristic_fallback(legal_actions, engine)
            logger.debug("Heuristic fallback: %s", fallback)
            return fallback

        return GameAction(ActionType.WAIT)

    def _parse_structured(self, llm_output: dict) -> Optional[GameAction]:
        """Parse from separate action and direction JSON fields."""
        action_str = llm_output.get("action", "")
        if not isinstance(action_str, str):
            return None

        if " " in action_str.strip():
            return self._parse_action_string(action_str)

        action_type = ActionType.from_string(action_str)
        if action_type is None:
            return None

        direction = None
        dir_str = llm_output.get("direction")
        if dir_str and isinstance(dir_str, str):
            direction = Direction.from_string(dir_str)

        return GameAction(action_type, direction)

    def _parse_action_string(self, action_str: str) -> Optional[GameAction]:
        """Parse from a combined string like 'MOVE NORTH'."""
        match = self._parse_pattern.search(action_str)
        if not match:
            return None

        action_type = ActionType.from_string(match.group(1))
        if action_type is None:
            return None

        direction = None
        if match.group(2):
            direction = Direction.from_string(match.group(2))

        return GameAction(action_type, direction)

    @staticmethod
    def _is_legal(action: GameAction, legal_actions: list[GameAction]) -> bool:
        """Check if an action exactly matches any legal action."""
        for legal in legal_actions:
            if (legal.action_type == action.action_type
                    and legal.direction == action.direction):
                return True
        return False

    @staticmethod
    def _find_closest_legal(action: GameAction,
                            legal_actions: list[GameAction]) -> Optional[GameAction]:
        """Find a legal action with the same action type but any direction."""
        same_type = [a for a in legal_actions if a.action_type == action.action_type]
        if same_type:
            return same_type[0]
        return None

    @staticmethod
    def _heuristic_fallback(legal_actions: list[GameAction],
                            engine: GameEngine) -> GameAction:
        """Select the highest-priority legal action by type ordering."""
        priority = {ActionType.ATTACK: 0, ActionType.HEAL: 1,
                    ActionType.MOVE: 2, ActionType.WAIT: 3}
        sorted_actions = sorted(legal_actions,
                                key=lambda a: priority.get(a.action_type, 99))
        return sorted_actions[0]
