"""Adaptive companion memory module.

Tracks HP trends, damage patterns, and action history over a sliding window
to dynamically shift the companion's behavioral persona between cautious,
aggressive, and balanced modes.
"""

import logging
from collections import deque
from typing import Optional

from game.engine import GameState
from game.entities import ActionType

logger = logging.getLogger(__name__)


class CompanionMemory:
    """Sliding-window memory that drives adaptive persona selection.

    Monitors recent game history to determine whether the companion
    should play cautiously, aggressively, or in a balanced manner.

    Args:
        window_size: Number of recent turns to track (default: 8).
    """

    PERSONA_AGGRESSIVE = "aggressive"
    PERSONA_CAUTIOUS = "cautious"
    PERSONA_BALANCED = "balanced"

    def __init__(self, window_size: int = 8) -> None:
        self.window_size = window_size
        self._hp_history: deque[int] = deque(maxlen=window_size)
        self._action_history: deque[ActionType] = deque(maxlen=window_size)
        self._damage_taken: deque[int] = deque(maxlen=window_size)
        self._enemies_killed: int = 0
        self._heals_used: int = 0
        self._persona: str = self.PERSONA_BALANCED
        self._last_hp: Optional[int] = None

    @property
    def persona(self) -> str:
        """Current behavioral persona (aggressive/cautious/balanced)."""
        return self._persona

    def update(self, state: GameState, action_type: ActionType) -> None:
        """Record a turn's outcome and recalculate persona.

        Args:
            state: Game state after the turn was executed.
            action_type: The action type that was taken this turn.
        """
        hp = state.player.hp
        self._hp_history.append(hp)
        self._action_history.append(action_type)

        if self._last_hp is not None:
            delta = self._last_hp - hp
            self._damage_taken.append(max(0, delta))
        else:
            self._damage_taken.append(0)
        self._last_hp = hp

        if action_type == ActionType.ATTACK:
            dead_count = sum(1 for e in state.enemies if not e.alive)
            if dead_count > self._enemies_killed:
                self._enemies_killed = dead_count
        if action_type == ActionType.HEAL:
            self._heals_used += 1

        self._update_persona(state)

    def _update_persona(self, state: GameState) -> None:
        """Recalculate persona based on recent HP, damage, and action ratios."""
        if len(self._hp_history) < 3:
            self._persona = self.PERSONA_BALANCED
            return

        avg_hp_ratio = sum(self._hp_history) / (len(self._hp_history) * state.player.max_hp)
        recent_damage = sum(self._damage_taken)
        attack_ratio = sum(1 for a in self._action_history if a == ActionType.ATTACK) / len(self._action_history)

        if avg_hp_ratio < 0.4 or recent_damage > state.player.max_hp * 0.5:
            self._persona = self.PERSONA_CAUTIOUS
        elif attack_ratio > 0.4 and avg_hp_ratio > 0.6:
            self._persona = self.PERSONA_AGGRESSIVE
        else:
            self._persona = self.PERSONA_BALANCED

        logger.debug("Persona updated: %s (avg_hp=%.2f, atk_ratio=%.2f)",
                     self._persona, avg_hp_ratio, attack_ratio)

    def get_context(self, state: GameState) -> str:
        """Build a memory context string for the LLM reasoning prompt.

        Args:
            state: Current game state for HP ratio calculation.

        Returns:
            Multi-line context string with persona, trends, and policy.
        """
        lines = []

        lines.append(f"Companion persona: {self._persona}")

        if len(self._hp_history) >= 3:
            recent = list(self._hp_history)[-3:]
            if recent[-1] < recent[0]:
                lines.append("HP trend: DECLINING (be more cautious)")
            elif recent[-1] > recent[0]:
                lines.append("HP trend: RECOVERING (can be more aggressive)")
            else:
                lines.append("HP trend: STABLE")

        if self._damage_taken:
            avg_dmg = sum(self._damage_taken) / len(self._damage_taken)
            if avg_dmg > 10:
                lines.append(f"Under heavy fire (avg {avg_dmg:.0f} dmg/turn)")

        lines.append(f"Enemies killed so far: {self._enemies_killed}")
        lines.append(f"Heals used: {self._heals_used}")

        if self._persona == self.PERSONA_CAUTIOUS:
            lines.append("POLICY: Prioritize survival. Avoid fights unless necessary. Seek health packs.")
        elif self._persona == self.PERSONA_AGGRESSIVE:
            lines.append("POLICY: Clear enemies aggressively. Push toward goal after threats eliminated.")
        else:
            lines.append("POLICY: Balance combat and movement. Engage when advantageous, retreat when weak.")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all memory and reset persona to balanced."""
        self._hp_history.clear()
        self._action_history.clear()
        self._damage_taken.clear()
        self._enemies_killed = 0
        self._heals_used = 0
        self._persona = self.PERSONA_BALANCED
        self._last_hp = None
