"""Game entity definitions for GridWorld Arena.

Defines all game objects (Player, Enemy, HealthPack, Goal) along with
enumerations for directions and action types used throughout the engine.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Direction(IntEnum):
    """Cardinal directions for movement and attack targeting."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    def to_delta(self) -> tuple[int, int]:
        """Convert direction to (dx, dy) grid offset."""
        _DELTAS = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
        return _DELTAS[self.value]

    @classmethod
    def from_string(cls, s: str) -> Optional["Direction"]:
        """Parse a direction from string (e.g. 'north', 'up', 'n')."""
        _MAP = {"north": 0, "south": 1, "east": 2, "west": 3,
                "up": 0, "down": 1, "right": 2, "left": 3,
                "n": 0, "s": 1, "e": 2, "w": 3}
        return cls(_MAP[s.lower()]) if s.lower() in _MAP else None


class ActionType(IntEnum):
    """Available action types for the companion agent."""

    MOVE = 0
    ATTACK = 1
    HEAL = 2
    WAIT = 3

    @classmethod
    def from_string(cls, s: str) -> Optional["ActionType"]:
        """Parse an action type from string (e.g. 'move', 'attack')."""
        _MAP = {"move": 0, "attack": 1, "heal": 2, "wait": 3}
        return cls(_MAP[s.lower()]) if s.lower() in _MAP else None


@dataclass
class GameAction:
    """A concrete action to be executed in the game engine."""

    action_type: ActionType
    direction: Optional[Direction] = None

    def __repr__(self) -> str:
        if self.direction is not None:
            return f"{self.action_type.name} {self.direction.name}"
        return self.action_type.name


@dataclass
class Entity:
    """Base class for all positioned objects on the grid."""

    x: int
    y: int
    alive: bool = True

    @property
    def pos(self) -> tuple[int, int]:
        """Return (x, y) grid coordinates."""
        return (self.x, self.y)

    def distance_to(self, other: "Entity") -> int:
        """Manhattan distance to another entity."""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Player(Entity):
    """The player-controlled character on the grid."""

    hp: int = 100
    max_hp: int = 100
    attack_power: int = 25
    attack_range: int = 2
    move_points: int = 1

    @property
    def is_dead(self) -> bool:
        """Check if player HP has reached zero."""
        return self.hp <= 0

    def take_damage(self, amount: int) -> int:
        """Apply damage and return actual damage dealt."""
        actual = min(amount, self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.alive = False
        return actual

    def restore_hp(self, amount: int) -> int:
        """Restore HP up to max and return actual amount restored."""
        actual = min(amount, self.max_hp - self.hp)
        self.hp += actual
        return actual


@dataclass
class Enemy(Entity):
    """A hostile NPC that patrols and chases the player."""

    hp: int = 100
    max_hp: int = 100
    attack_power: int = 15
    detection_range: int = 3
    patrol_direction: Direction = Direction.EAST
    entity_id: int = 0

    @property
    def is_dead(self) -> bool:
        """Check if enemy HP has reached zero."""
        return self.hp <= 0

    def take_damage(self, amount: int) -> int:
        """Apply damage and return actual damage dealt."""
        actual = min(amount, self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.alive = False
        return actual

    def can_detect(self, target: Entity) -> bool:
        """Check if target is within detection range."""
        return self.distance_to(target) <= self.detection_range


@dataclass
class HealthPack(Entity):
    """A consumable item that restores player HP."""

    restore_amount: int = 30
    consumed: bool = False

    def consume(self) -> int:
        """Consume the health pack and return restore amount (0 if already used)."""
        if not self.consumed:
            self.consumed = True
            self.alive = False
            return self.restore_amount
        return 0


@dataclass
class Goal(Entity):
    """The objective tile the player must reach to win."""

    reached: bool = False
