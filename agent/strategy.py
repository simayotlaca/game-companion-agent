"""Receding horizon tactical planning module.

Generates 3-5 step tactical plans based on current game state,
re-evaluated every turn to avoid committing to stale strategies.
"""

import logging
from typing import Optional

from game.engine import GameState
from game.entities import ActionType

logger = logging.getLogger(__name__)


class TacticalPlan:
    """A multi-step tactical plan with objective, steps, and fallback.

    Attributes:
        steps: Ordered list of planned actions (3-5 items).
        current_step: Index of the step currently being executed.
        objective: High-level goal of this plan.
        risk_level: Assessed risk ('low', 'medium', 'high').
        next_best: Fallback action if the plan cannot be executed.
    """

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.current_step: int = 0
        self.objective: str = ""
        self.risk_level: str = "low"
        self.next_best: str = ""

    def to_dict(self) -> dict:
        """Serialize plan to a dictionary."""
        return {
            "objective": self.objective,
            "steps": self.steps,
            "current_step": self.current_step,
            "risk_level": self.risk_level,
            "next_best": self.next_best,
        }

    @staticmethod
    def from_dict(data: dict) -> "TacticalPlan":
        """Deserialize a plan from a dictionary."""
        plan = TacticalPlan()
        plan.objective = data.get("objective", "")
        plan.steps = data.get("steps", [])
        plan.current_step = data.get("current_step", 0)
        plan.risk_level = data.get("risk_level", "low")
        plan.next_best = data.get("next_best", "")
        return plan

    def __repr__(self) -> str:
        step_display = []
        for i, step in enumerate(self.steps):
            marker = ">>>" if i == self.current_step else "   "
            step_display.append(f"  {marker} {i+1}. {step}")
        return (
            f"Plan: {self.objective} (risk: {self.risk_level})\n"
            + "\n".join(step_display)
            + f"\n  Fallback: {self.next_best}"
        )


class StrategyModule:
    """Generates and manages receding-horizon tactical plans.

    Each turn, a new plan is generated based on the current state.
    Plans are 3-5 steps long and categorized by objective type:
    survive/heal, engage enemy, tactical retreat, or navigate to goal.
    """

    def __init__(self) -> None:
        self._current_plan: Optional[TacticalPlan] = None

    @property
    def current_plan(self) -> Optional[TacticalPlan]:
        """The most recently generated tactical plan."""
        return self._current_plan

    def generate_plan(self, state: GameState) -> TacticalPlan:
        """Generate a new tactical plan based on current game state.

        Args:
            state: Current game state with entity positions and HP.

        Returns:
            A TacticalPlan with objective, steps, risk level, and fallback.
        """
        plan = TacticalPlan()
        p = state.player
        enemies = state.living_enemies
        health_packs = state.available_health_packs

        hp_ratio = p.hp / p.max_hp
        nearby_enemies = [e for e in enemies if p.distance_to(e) <= p.attack_range + 2]
        threats_in_range = [e for e in enemies if p.distance_to(e) <= p.attack_range]
        goal_dist = p.distance_to(state.goal)

        if hp_ratio < 0.3 and nearby_enemies:
            plan.risk_level = "high"
        elif hp_ratio < 0.5 or len(nearby_enemies) >= 2:
            plan.risk_level = "medium"
        else:
            plan.risk_level = "low"

        if hp_ratio < 0.3 and health_packs:
            closest_hp = min(health_packs, key=lambda h: p.distance_to(h))
            plan.objective = "Survive and heal"
            plan.steps = [
                f"Move toward health pack at ({closest_hp.x},{closest_hp.y})",
                "Use health pack to restore HP",
                "Reassess threats after healing",
                "Resume objective pursuit",
            ]
            plan.next_best = "Retreat from enemies if no health packs available"

        elif threats_in_range and hp_ratio > 0.4:
            target = min(threats_in_range, key=lambda e: e.hp)
            plan.objective = f"Eliminate enemy #{target.entity_id}"
            plan.steps = [
                f"Attack enemy #{target.entity_id} (HP: {target.hp}/{target.max_hp})",
                "Continue attacking if enemy survives",
                "Check for additional threats",
                "Move toward goal after area is clear",
            ]
            plan.next_best = "Disengage and move toward goal if HP drops below 30%"

        elif nearby_enemies and hp_ratio <= 0.4:
            plan.objective = "Tactical retreat"
            plan.steps = [
                "Move away from nearest enemy",
                "Find health pack or safe path",
                "Heal if possible",
                "Re-engage or proceed to goal",
            ]
            plan.next_best = "WAIT if all escape routes blocked"

        else:
            plan.objective = "Navigate to goal"
            plan.steps = [
                f"Move toward goal at ({state.goal.x},{state.goal.y}) — distance: {goal_dist}",
                "Avoid enemy detection zones",
                "Pick up health packs on the way",
                "Engage enemies only if blocking path",
                f"Reach goal tile ({state.goal.x},{state.goal.y})",
            ]
            plan.next_best = "Engage blocking enemies if unavoidable"

        self._current_plan = plan
        logger.debug("Plan generated: %s (risk: %s)", plan.objective, plan.risk_level)
        return plan

    def get_plan_context(self, state: GameState) -> str:
        """Generate a plan and return it as a context string for the LLM.

        Args:
            state: Current game state.

        Returns:
            Formatted multi-line plan string.
        """
        plan = self.generate_plan(state)
        lines = [
            f"Current plan: {plan.objective}",
            f"Risk level: {plan.risk_level}",
            "Steps:",
        ]
        for i, step in enumerate(plan.steps):
            marker = ">>>" if i == 0 else "   "
            lines.append(f"  {marker} {i+1}. {step}")
        lines.append(f"Fallback: {plan.next_best}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear the current plan."""
        self._current_plan = None
