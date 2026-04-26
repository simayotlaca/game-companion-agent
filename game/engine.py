"""Core game engine for GridWorld Arena.

Manages game state, turn processing, entity interactions, and
legal action enumeration for the companion agent.
"""

import logging
import random
from typing import Optional

from config import GameConfig
from game.entities import (
    Player, Enemy, HealthPack, Goal, GameAction,
    ActionType, Direction, Entity
)

logger = logging.getLogger(__name__)


class GameState:
    """Read-only snapshot of the current game state.

    This object is a snapshot — do NOT mutate its fields after creation.
    The engine owns all mutable state; GameState is for observation only.

    Attributes:
        player: The player entity with position and HP.
        enemies: All enemy entities (alive or dead).
        health_packs: All health pack entities (consumed or available).
        goal: The objective tile.
        turn: Current turn number (>= 0).
        grid_width: Width of the game grid (>= 4).
        grid_height: Height of the game grid (>= 4).
        game_over: Whether the game has ended.
        won: Whether the player won.
        last_action: The most recent action taken.
        last_result: Human-readable result of the last action.
    """

    def __init__(self, player: Player, enemies: list[Enemy],
                 health_packs: list[HealthPack], goal: Goal,
                 turn: int, grid_w: int, grid_h: int,
                 game_over: bool = False, won: bool = False,
                 last_action: Optional[GameAction] = None,
                 last_result: str = "") -> None:
        self.player = player
        self.enemies = enemies
        self.health_packs = health_packs
        self.goal = goal
        self.turn = turn
        self.grid_width = grid_w
        self.grid_height = grid_h
        self.game_over = game_over
        self.won = won
        self.last_action = last_action
        self.last_result = last_result

    @property
    def living_enemies(self) -> list[Enemy]:
        """Return list of enemies that are still alive."""
        return [e for e in self.enemies if e.alive]

    @property
    def available_health_packs(self) -> list[HealthPack]:
        """Return list of health packs that have not been consumed."""
        return [h for h in self.health_packs if not h.consumed]

    def to_text(self) -> str:
        """Serialize game state to a human-readable text format."""
        lines = [
            f"Turn: {self.turn}",
            f"Grid: {self.grid_width}x{self.grid_height}",
            f"Player: pos=({self.player.x},{self.player.y}) HP={self.player.hp}/{self.player.max_hp}",
        ]
        for e in self.living_enemies:
            dist = self.player.distance_to(e)
            lines.append(
                f"Enemy #{e.entity_id}: pos=({e.x},{e.y}) HP={e.hp}/{e.max_hp} distance={dist}"
            )
        for h in self.available_health_packs:
            dist = self.player.distance_to(h)
            lines.append(f"HealthPack: pos=({h.x},{h.y}) restores={h.restore_amount} distance={dist}")
        lines.append(f"Goal: pos=({self.goal.x},{self.goal.y}) distance={self.player.distance_to(self.goal)}")
        return "\n".join(lines)


class GameEngine:
    """Turn-based game engine that processes actions and updates state.

    The engine manages entity placement, action validation, combat resolution,
    enemy AI (patrol + chase), and win/loss conditions.

    Args:
        config: Game configuration parameters (grid size, entity counts, etc.).
    """

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

        self._player: Optional[Player] = None
        self._enemies: list[Enemy] = []
        self._health_packs: list[HealthPack] = []
        self._goal: Optional[Goal] = None
        self._turn: int = 0
        self._game_over: bool = False
        self._won: bool = False
        self._last_result: str = ""

        self._occupied: set[tuple[int, int]] = set()

    def reset(self) -> GameState:
        """Initialize a new game with randomized entity placement.

        Returns:
            The initial GameState snapshot.
        """
        self._occupied.clear()
        self._turn = 0
        self._game_over = False
        self._won = False

        self._player = Player(
            x=self.rng.randint(0, 1),
            y=self.rng.randint(0, 1),
            hp=self.config.player_hp,
            max_hp=self.config.player_hp,
            attack_power=self.config.player_attack,
            attack_range=self.config.player_attack_range,
            move_points=self.config.player_move_points,
        )
        self._occupied.add(self._player.pos)

        gx = self.rng.randint(self.config.grid_width - 2, self.config.grid_width - 1)
        gy = self.rng.randint(self.config.grid_height - 2, self.config.grid_height - 1)
        self._goal = Goal(x=gx, y=gy)
        self._occupied.add(self._goal.pos)

        self._enemies = []
        for i in range(self.config.num_enemies):
            pos = self._random_free_pos()
            enemy = Enemy(
                x=pos[0], y=pos[1],
                hp=self.config.enemy_hp,
                max_hp=self.config.enemy_hp,
                attack_power=self.config.enemy_attack,
                detection_range=self.config.enemy_detection_range,
                patrol_direction=Direction(self.rng.randint(0, 3)),
                entity_id=i,
            )
            self._enemies.append(enemy)
            self._occupied.add(pos)

        self._health_packs = []
        for _ in range(self.config.num_health_packs):
            pos = self._random_free_pos()
            hp_item = HealthPack(
                x=pos[0], y=pos[1],
                restore_amount=self.config.health_pack_restore,
            )
            self._health_packs.append(hp_item)
            self._occupied.add(pos)

        self._last_result = "Game started."
        logger.info("Game initialized: %dx%d grid, %d enemies, goal at (%d,%d)",
                     self.config.grid_width, self.config.grid_height,
                     self.config.num_enemies, gx, gy)
        return self._snapshot()

    def step(self, action: GameAction) -> GameState:
        """Process one game turn: player action, then enemy turns.

        Contract:
            - Exactly one action is processed per call.
            - Player HP stays in [0, max_hp] after every step.
            - Turn counter increments by exactly 1.
            - Returns a snapshot; caller must not mutate the returned state.

        Args:
            action: The action to execute this turn.

        Returns:
            Updated GameState snapshot after the turn.
        """
        if self._game_over:
            return self._snapshot()

        assert isinstance(action, GameAction), f"action must be GameAction, got {type(action)}"
        turn_before = self._turn

        self._turn += 1
        result_parts = []

        player_result = self._process_player_action(action)
        result_parts.append(player_result)

        if self._player.pos == self._goal.pos:
            self._won = True
            self._game_over = True
            self._last_result = " ".join(result_parts) + " Player reached the goal! Victory!"
            logger.info("Victory on turn %d", self._turn)
            return self._snapshot(action)

        if not self._game_over:
            for enemy in self._enemies:
                if not enemy.alive:
                    continue
                enemy_result = self._process_enemy_turn(enemy)
                if enemy_result:
                    result_parts.append(enemy_result)

        if self._player.is_dead:
            self._game_over = True
            result_parts.append("Player has been defeated!")
            logger.info("Defeat on turn %d", self._turn)

        if self._turn >= self.config.max_turns:
            self._game_over = True
            result_parts.append("Turn limit reached.")

        self._last_result = " ".join(result_parts)

        # Post-step invariants
        assert self._turn == turn_before + 1, "Turn counter must increment by exactly 1"
        assert 0 <= self._player.hp <= self._player.max_hp, (
            f"Player HP out of range: {self._player.hp}/{self._player.max_hp}"
        )
        assert self._in_bounds(self._player.x, self._player.y), (
            f"Player out of bounds: ({self._player.x},{self._player.y})"
        )

        return self._snapshot(action)

    def get_legal_actions(self) -> list[GameAction]:
        """Enumerate all valid actions for the current state.

        Returns:
            List of legal GameAction objects (always includes WAIT).
        """
        actions = [GameAction(ActionType.WAIT)]

        for d in Direction:
            dx, dy = d.to_delta()
            nx, ny = self._player.x + dx, self._player.y + dy
            if self._in_bounds(nx, ny) and not self._is_enemy_at(nx, ny):
                actions.append(GameAction(ActionType.MOVE, d))

        for enemy in self._enemies:
            if enemy.alive and self._player.distance_to(enemy) <= self._player.attack_range:
                dx = enemy.x - self._player.x
                dy = enemy.y - self._player.y
                if abs(dx) >= abs(dy):
                    d = Direction.EAST if dx > 0 else Direction.WEST
                else:
                    d = Direction.SOUTH if dy > 0 else Direction.NORTH
                actions.append(GameAction(ActionType.ATTACK, d))

        for hp_item in self._health_packs:
            if (not hp_item.consumed
                    and self._player.distance_to(hp_item) <= 1
                    and self._player.hp < self._player.max_hp):
                actions.append(GameAction(ActionType.HEAL))
                break

        return actions

    def _process_player_action(self, action: GameAction) -> str:
        """Dispatch and execute the player's chosen action."""
        if action.action_type == ActionType.MOVE:
            return self._player_move(action.direction)
        elif action.action_type == ActionType.ATTACK:
            return self._player_attack(action.direction)
        elif action.action_type == ActionType.HEAL:
            return self._player_heal()
        else:
            return "Player waits."

    def _player_move(self, direction: Optional[Direction]) -> str:
        """Move the player one tile in the given direction."""
        if direction is None:
            return "Invalid move (no direction)."
        dx, dy = direction.to_delta()
        nx, ny = self._player.x + dx, self._player.y + dy
        if not self._in_bounds(nx, ny):
            return f"Cannot move {direction.name} — out of bounds."
        if self._is_enemy_at(nx, ny):
            return f"Cannot move {direction.name} — blocked by enemy."
        self._player.x, self._player.y = nx, ny

        for hp_item in self._health_packs:
            if not hp_item.consumed and hp_item.pos == (nx, ny):
                restored = hp_item.consume()
                self._player.restore_hp(restored)
                return (f"Moved {direction.name} to ({nx},{ny}). "
                        f"Picked up health pack! +{restored} HP (now {self._player.hp}).")

        return f"Moved {direction.name} to ({nx},{ny})."

    def _player_attack(self, direction: Optional[Direction]) -> str:
        """Attack the nearest enemy in the given direction within range."""
        if direction is None:
            return "Invalid attack (no direction)."

        target = None
        min_dist = float("inf")
        for enemy in self._enemies:
            if not enemy.alive:
                continue
            if self._player.distance_to(enemy) > self._player.attack_range:
                continue
            dx = enemy.x - self._player.x
            dy = enemy.y - self._player.y
            if direction == Direction.EAST and dx > 0:
                pass
            elif direction == Direction.WEST and dx < 0:
                pass
            elif direction == Direction.SOUTH and dy > 0:
                pass
            elif direction == Direction.NORTH and dy < 0:
                pass
            else:
                continue
            dist = self._player.distance_to(enemy)
            if dist < min_dist:
                min_dist = dist
                target = enemy

        if target is None:
            return f"Attack {direction.name} — no enemy in range."

        dmg = target.take_damage(self._player.attack_power)
        if target.is_dead:
            return (f"Attacked enemy #{target.entity_id} {direction.name} for {dmg} damage. "
                    f"Enemy defeated!")
        return (f"Attacked enemy #{target.entity_id} {direction.name} for {dmg} damage. "
                f"Enemy HP: {target.hp}/{target.max_hp}.")

    def _player_heal(self) -> str:
        """Use the nearest adjacent health pack."""
        for hp_item in self._health_packs:
            if not hp_item.consumed and self._player.distance_to(hp_item) <= 1:
                restored = hp_item.consume()
                actual = self._player.restore_hp(restored)
                return f"Used health pack. +{actual} HP (now {self._player.hp}/{self._player.max_hp})."
        return "No health pack nearby."

    def _process_enemy_turn(self, enemy: Enemy) -> str:
        """Execute one enemy's turn: chase player or patrol."""
        if enemy.can_detect(self._player):
            if enemy.distance_to(self._player) <= 1:
                dmg = self._player.take_damage(enemy.attack_power)
                return f"Enemy #{enemy.entity_id} attacks player for {dmg} damage! Player HP: {self._player.hp}."
            dx = self._player.x - enemy.x
            dy = self._player.y - enemy.y
            if abs(dx) >= abs(dy):
                move_x = 1 if dx > 0 else -1
                nx, ny = enemy.x + move_x, enemy.y
            else:
                move_y = 1 if dy > 0 else -1
                nx, ny = enemy.x, enemy.y + move_y
            if self._in_bounds(nx, ny) and not self._is_entity_at(nx, ny, exclude=enemy):
                enemy.x, enemy.y = nx, ny
            return f"Enemy #{enemy.entity_id} chases player to ({enemy.x},{enemy.y})."
        else:
            dx, dy = enemy.patrol_direction.to_delta()
            nx, ny = enemy.x + dx, enemy.y + dy
            if self._in_bounds(nx, ny) and not self._is_entity_at(nx, ny, exclude=enemy):
                enemy.x, enemy.y = nx, ny
            else:
                enemy.patrol_direction = Direction(self.rng.randint(0, 3))
            return ""

    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid boundaries."""
        return 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height

    def _is_enemy_at(self, x: int, y: int) -> bool:
        """Check if a living enemy occupies the given tile."""
        return any(e.alive and e.x == x and e.y == y for e in self._enemies)

    def _is_entity_at(self, x: int, y: int, exclude: Optional[Entity] = None) -> bool:
        """Check if any entity (player or enemy) occupies the given tile."""
        if self._player.pos == (x, y) and self._player is not exclude:
            return True
        for e in self._enemies:
            if e.alive and e.pos == (x, y) and e is not exclude:
                return True
        return False

    def _random_free_pos(self) -> tuple[int, int]:
        """Find an unoccupied grid position.

        Raises:
            RuntimeError: If no free position exists on the grid.
        """
        for _ in range(200):
            x = self.rng.randint(0, self.config.grid_width - 1)
            y = self.rng.randint(0, self.config.grid_height - 1)
            if (x, y) not in self._occupied:
                self._occupied.add((x, y))
                return (x, y)
        for x in range(self.config.grid_width):
            for y in range(self.config.grid_height):
                if (x, y) not in self._occupied:
                    self._occupied.add((x, y))
                    return (x, y)
        raise RuntimeError("No free position on grid")

    def _snapshot(self, action: Optional[GameAction] = None) -> GameState:
        """Create a GameState snapshot of the current engine state."""
        return GameState(
            player=self._player,
            enemies=list(self._enemies),
            health_packs=list(self._health_packs),
            goal=self._goal,
            turn=self._turn,
            grid_w=self.config.grid_width,
            grid_h=self.config.grid_height,
            game_over=self._game_over,
            won=self._won,
            last_action=action,
            last_result=self._last_result,
        )
