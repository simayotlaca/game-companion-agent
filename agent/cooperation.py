from typing import Optional, Callable
from game.engine import GameState
from game.entities import ActionType


class PlayerCommand:

    COMMANDS = {
        "focus heal": {"override_action": "HEAL", "description": "Prioritize healing this turn"},
        "heal": {"override_action": "HEAL", "description": "Prioritize healing this turn"},
        "rush": {"intent_override": "rush goal", "description": "Sprint toward the goal"},
        "rush goal": {"intent_override": "rush goal", "description": "Sprint toward the goal"},
        "hold": {"override_action": "WAIT", "description": "Hold position this turn"},
        "hold position": {"override_action": "WAIT", "description": "Hold position this turn"},
        "wait": {"override_action": "WAIT", "description": "Wait this turn"},
        "attack": {"intent_override": "aggressive", "description": "Focus on attacking"},
        "fight": {"intent_override": "aggressive", "description": "Focus on attacking"},
        "retreat": {"intent_override": "cautious", "description": "Fall back from enemies"},
        "fall back": {"intent_override": "cautious", "description": "Fall back from enemies"},
        "ignore enemies": {"intent_override": "rush goal", "description": "Skip fights, go to goal"},
    }

    def __init__(self, raw_input: str):
        self.raw = raw_input.strip().lower()
        self.override_action: Optional[str] = None
        self.intent_override: Optional[str] = None
        self.description: str = ""
        self.is_valid: bool = False
        self._parse()

    def _parse(self):
        if not self.raw:
            return

        if self.raw in self.COMMANDS:
            cmd = self.COMMANDS[self.raw]
            self.override_action = cmd.get("override_action")
            self.intent_override = cmd.get("intent_override")
            self.description = cmd.get("description", "")
            self.is_valid = True
            return

        for key, cmd in self.COMMANDS.items():
            if key in self.raw or self.raw in key:
                self.override_action = cmd.get("override_action")
                self.intent_override = cmd.get("intent_override")
                self.description = cmd.get("description", "")
                self.is_valid = True
                return

    def __repr__(self) -> str:
        if not self.is_valid:
            return f"PlayerCommand('{self.raw}', invalid)"
        parts = [f"PlayerCommand('{self.raw}'"]
        if self.override_action:
            parts.append(f"override={self.override_action}")
        if self.intent_override:
            parts.append(f"intent={self.intent_override}")
        return ", ".join(parts) + ")"


class PlanProposal:

    def __init__(self, objective: str, steps: list[str],
                 risk_level: str, suggested_action: str,
                 companion_rationale: str):
        self.objective = objective
        self.steps = steps
        self.risk_level = risk_level
        self.suggested_action = suggested_action
        self.companion_rationale = companion_rationale
        self.player_approved: bool = False
        self.player_modification: Optional[str] = None

    def present(self) -> str:
        lines = [
            f"[Companion proposes] {self.objective}",
            f"  Suggested action: {self.suggested_action}",
            f"  Risk: {self.risk_level}",
            f"  Rationale: {self.companion_rationale}",
            "  Plan:",
        ]
        for i, step in enumerate(self.steps[:4]):
            marker = ">>>" if i == 0 else "   "
            lines.append(f"    {marker} {i+1}. {step}")
        lines.append("  [Approve / Override / Command?]")
        return "\n".join(lines)

    def approve(self):
        self.player_approved = True

    def modify(self, modification: str):
        self.player_modification = modification
        self.player_approved = True


class RoleAdvisor:

    @staticmethod
    def suggest_role(state: GameState, companion_action: str,
                     companion_direction: Optional[str] = None) -> str:
        p = state.player
        enemies = state.living_enemies
        hp_ratio = p.hp / p.max_hp
        goal_dist = p.distance_to(state.goal)

        if companion_action == "ATTACK" and companion_direction:
            if len(enemies) > 1:
                other_dirs = []
                for e in enemies:
                    if e.alive:
                        dx = e.x - p.x
                        dy = e.y - p.y
                        if abs(dx) >= abs(dy):
                            d = "east" if dx > 0 else "west"
                        else:
                            d = "south" if dy > 0 else "north"
                        if d != companion_direction.lower():
                            other_dirs.append(d)
                if other_dirs:
                    return (f"I'm engaging the enemy to the {companion_direction.lower()}. "
                            f"Watch your {other_dirs[0]} side!")
            return (f"I'll handle the enemy to the {companion_direction.lower()} — "
                    f"you push toward the goal!")

        elif companion_action == "HEAL":
            if enemies:
                nearest = min(enemies, key=lambda e: p.distance_to(e))
                dx = nearest.x - p.x
                dy = nearest.y - p.y
                if abs(dx) >= abs(dy):
                    threat_dir = "east" if dx > 0 else "west"
                else:
                    threat_dir = "south" if dy > 0 else "north"
                return (f"I'm healing up — cover the {threat_dir} "
                        f"while I recover!")
            return "I'm healing up — keep moving toward the objective."

        elif companion_action == "MOVE":
            if goal_dist <= 3:
                return "Almost there! I'll clear the path — get ready to reach the goal."
            if hp_ratio < 0.4:
                return "I'm falling back to regroup. Let's find a health pack."
            if enemies and all(p.distance_to(e) > 4 for e in enemies):
                return "Coast is clear — let's push together toward the objective."
            return f"Moving toward the goal. Stay close, {goal_dist} tiles to go."

        elif companion_action == "WAIT":
            if enemies:
                return "Holding position. Let's wait for the enemy to move first."
            return "Holding position — assessing the situation."

        return "Let's coordinate our next move."


class CooperationController:

    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        self._current_proposal: Optional[PlanProposal] = None
        self._current_command: Optional[PlayerCommand] = None
        self._role_advisor = RoleAdvisor()
        self._player_input_callback: Optional[Callable] = None

    @property
    def current_command(self) -> Optional[PlayerCommand]:
        return self._current_command

    @property
    def current_proposal(self) -> Optional[PlanProposal]:
        return self._current_proposal

    def set_interactive(self, callback: Callable[[str], str]):
        self._auto_approve = False
        self._player_input_callback = callback

    def propose_plan(self, state: GameState, plan_objective: str,
                     plan_steps: list[str], risk_level: str,
                     suggested_action: str, rationale: str) -> PlanProposal:
        proposal = PlanProposal(
            objective=plan_objective,
            steps=plan_steps,
            risk_level=risk_level,
            suggested_action=suggested_action,
            companion_rationale=rationale,
        )

        if self._auto_approve:
            proposal.approve()
        elif self._player_input_callback:
            player_response = self._player_input_callback(proposal.present())
            self._process_player_response(proposal, player_response)
        else:
            proposal.approve()

        self._current_proposal = proposal
        return proposal

    def receive_player_input(self, raw_input: str):
        cmd = PlayerCommand(raw_input)
        self._current_command = cmd if cmd.is_valid else None

    def clear_turn_command(self):
        self._current_command = None

    def get_role_message(self, state: GameState, action: str,
                         direction: Optional[str] = None) -> str:
        return self._role_advisor.suggest_role(state, action, direction)

    def get_turn_context(self) -> str:
        parts = []
        if self._current_command and self._current_command.is_valid:
            parts.append(f"Player command: {self._current_command.description}")
            if self._current_command.override_action:
                parts.append(f"Action override: {self._current_command.override_action}")
            if self._current_command.intent_override:
                parts.append(f"Intent override: {self._current_command.intent_override}")

        if self._current_proposal and self._current_proposal.player_modification:
            parts.append(f"Player modified plan: {self._current_proposal.player_modification}")

        return "\n".join(parts) if parts else ""

    def _process_player_response(self, proposal: PlanProposal, response: str):
        response = response.strip().lower()

        if not response or response in ("ok", "yes", "approve", "go", "y"):
            proposal.approve()
        else:
            cmd = PlayerCommand(response)
            if cmd.is_valid:
                self._current_command = cmd
                proposal.modify(response)
            else:
                proposal.modify(response)

    def reset(self):
        self._current_proposal = None
        self._current_command = None
