"""Performance metrics logging for game runs.

Tracks per-turn latencies (VLM, LLM, parse), fallback rates,
action distributions, and aggregate game results across sessions.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Per-turn performance data."""

    turn: int = 0
    vlm_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    parse_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    action_was_fallback: bool = False
    action_type: str = ""


class MetricsLogger:
    """Collects and reports performance metrics across game turns.

    Tracks VLM/LLM latency, fallback rates, action distributions,
    and win rates for deployment monitoring and optimization.
    """

    def __init__(self) -> None:
        self._turns: list[TurnMetrics] = []
        self._current: Optional[TurnMetrics] = None
        self._timer_stack: dict[str, float] = {}
        self._game_results: list[dict] = []

    def start_turn(self, turn: int) -> None:
        """Begin timing a new turn."""
        self._current = TurnMetrics(turn=turn)
        self._timer_stack["turn"] = time.perf_counter()

    def record_vlm_start(self) -> None:
        """Mark the start of VLM perception."""
        self._timer_stack["vlm"] = time.perf_counter()

    def record_vlm_end(self) -> None:
        """Mark the end of VLM perception and record latency."""
        if "vlm" in self._timer_stack and self._current:
            self._current.vlm_latency_ms = (time.perf_counter() - self._timer_stack["vlm"]) * 1000

    def record_llm_start(self) -> None:
        """Mark the start of LLM reasoning."""
        self._timer_stack["llm"] = time.perf_counter()

    def record_llm_end(self) -> None:
        """Mark the end of LLM reasoning and record latency."""
        if "llm" in self._timer_stack and self._current:
            self._current.llm_latency_ms = (time.perf_counter() - self._timer_stack["llm"]) * 1000

    def record_parse_start(self) -> None:
        """Mark the start of action parsing."""
        self._timer_stack["parse"] = time.perf_counter()

    def record_parse_end(self) -> None:
        """Mark the end of action parsing and record latency."""
        if "parse" in self._timer_stack and self._current:
            self._current.parse_latency_ms = (time.perf_counter() - self._timer_stack["parse"]) * 1000

    def end_turn(self, action_type: str, was_fallback: bool) -> None:
        """Finalize the current turn's metrics.

        Args:
            action_type: The action type name taken this turn.
            was_fallback: Whether the action required fallback parsing.
        """
        if self._current:
            self._current.total_latency_ms = (time.perf_counter() - self._timer_stack.get("turn", time.perf_counter())) * 1000
            self._current.action_was_fallback = was_fallback
            self._current.action_type = action_type
            self._turns.append(self._current)
            self._current = None

    def record_game_result(self, won: bool, total_turns: int,
                           final_hp: int, enemies_defeated: int,
                           total_enemies: int) -> None:
        """Record the outcome of a completed game.

        Args:
            won: Whether the player won.
            total_turns: Total turns played.
            final_hp: Player HP at game end.
            enemies_defeated: Number of enemies killed.
            total_enemies: Total number of enemies in the game.
        """
        self._game_results.append({
            "won": won,
            "turns": total_turns,
            "final_hp": final_hp,
            "enemies_defeated": enemies_defeated,
            "total_enemies": total_enemies,
        })

    @property
    def total_turns(self) -> int:
        """Total number of turns recorded."""
        return len(self._turns)

    @property
    def fallback_count(self) -> int:
        """Number of turns that required fallback action parsing."""
        return sum(1 for t in self._turns if t.action_was_fallback)

    @property
    def fallback_rate(self) -> float:
        """Fraction of turns that required fallback (0.0 to 1.0)."""
        return self.fallback_count / max(1, self.total_turns)

    @property
    def avg_vlm_latency_ms(self) -> float:
        """Average VLM perception latency in milliseconds."""
        vals = [t.vlm_latency_ms for t in self._turns if t.vlm_latency_ms > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_llm_latency_ms(self) -> float:
        """Average LLM reasoning latency in milliseconds."""
        vals = [t.llm_latency_ms for t in self._turns if t.llm_latency_ms > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_total_latency_ms(self) -> float:
        """Average total turn latency in milliseconds."""
        vals = [t.total_latency_ms for t in self._turns if t.total_latency_ms > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def action_distribution(self) -> dict[str, int]:
        """Count of each action type taken across all turns."""
        return dict(Counter(t.action_type for t in self._turns))

    @property
    def win_rate(self) -> float:
        """Win rate across all recorded games (0.0 to 1.0)."""
        if not self._game_results:
            return 0.0
        return sum(1 for g in self._game_results if g["won"]) / len(self._game_results)

    def summary(self) -> str:
        """Generate a formatted performance summary string."""
        lines = [
            "─── Performance Metrics ───",
            f"  Turns played:       {self.total_turns}",
            f"  Fallback actions:   {self.fallback_count}/{self.total_turns} ({self.fallback_rate:.1%})",
            f"  Action distribution: {self.action_distribution}",
            "",
            "  Latency (avg per turn):",
            f"    VLM perception:   {self.avg_vlm_latency_ms:.1f} ms",
            f"    LLM reasoning:    {self.avg_llm_latency_ms:.1f} ms",
            f"    Total turn:       {self.avg_total_latency_ms:.1f} ms",
        ]

        if self._game_results:
            lines.append("")
            lines.append(f"  Games played: {len(self._game_results)}")
            lines.append(f"  Win rate:     {self.win_rate:.1%}")
            avg_turns = sum(g["turns"] for g in self._game_results) / len(self._game_results)
            lines.append(f"  Avg turns:    {avg_turns:.1f}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear turn metrics for a new game (keeps game results)."""
        self._turns.clear()
        self._current = None
        self._timer_stack.clear()

    def reset_all(self) -> None:
        """Clear all metrics including game results."""
        self.reset()
        self._game_results.clear()
