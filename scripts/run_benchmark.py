"""
Benchmark runner for game-companion-agent.

Runs N offline games (no GPU needed) and produces aggregate metrics
suitable for portfolio reporting:
  - Win rate
  - Schema compliance rate (1 - fallback rate)
  - Avg / P95 / P99 turn latency
  - Action distribution
  - Enemies defeated rate
  - Avg turns per game

Usage:
    python run_benchmark.py              # 50 games, quiet
    python run_benchmark.py --games 100  # 100 games
    python run_benchmark.py --verbose    # show each turn
"""

import argparse
import json
import sys
import os
import time
from statistics import mean, quantiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.companion import CompanionAgent


def run_benchmark(n_games: int = 50, verbose: bool = False, seed_offset: int = 0):
    render_config = RenderConfig()

    # Per-game tracking
    wins = 0
    fallback_turns = 0
    total_turns = 0
    latencies_ms = []
    action_counts: dict[str, int] = {}
    enemies_defeated_total = 0
    enemies_total = 0
    turns_per_game = []

    print(f"\n{'='*60}")
    print(f"  BENCHMARK — {n_games} offline games")
    print(f"  Grid: 8x8 | Enemies: 3 | Max turns: 40")
    print(f"{'='*60}\n")

    start_wall = time.perf_counter()

    for i in range(n_games):
        seed = seed_offset + i  # different seed each game → diverse scenarios

        game_config = GameConfig(
            grid_width=8,
            grid_height=8,
            max_turns=40,
            num_enemies=3,
            num_health_packs=2,
            seed=seed,
        )
        agent_config = AgentConfig(seed=seed, deterministic_mode=True)

        engine = GameEngine(game_config)
        renderer = GameRenderer(render_config)
        agent = CompanionAgent(engine=engine, renderer=renderer,
                               config=agent_config, offline=True)

        final_state, logs = agent.play_game(max_turns=30, verbose=verbose)

        # ── Aggregate ──────────────────────────────────────────────
        if final_state.won:
            wins += 1

        game_turns = len(logs)
        turns_per_game.append(game_turns)
        total_turns += game_turns

        game_fallbacks = sum(1 for log in logs if log.action_was_fallback)
        fallback_turns += game_fallbacks

        for t in agent.metrics._turns:
            if t.total_latency_ms > 0:
                latencies_ms.append(t.total_latency_ms)
            if t.action_type:
                action_counts[t.action_type] = action_counts.get(t.action_type, 0) + 1

        defeated = sum(1 for e in final_state.enemies if not e.alive)
        enemies_defeated_total += defeated
        enemies_total += len(final_state.enemies)

        # Progress bar
        if not verbose:
            bar_len = 30
            filled = int(bar_len * (i + 1) / n_games)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {i+1}/{n_games}  wins:{wins}", end="", flush=True)

    wall_time = time.perf_counter() - start_wall
    print()  # newline after progress bar

    # ── Compute stats ──────────────────────────────────────────────
    win_rate = wins / n_games
    schema_compliance = 1.0 - (fallback_turns / max(1, total_turns))
    avg_latency = mean(latencies_ms) if latencies_ms else 0.0
    avg_turns = mean(turns_per_game)

    # P95 / P99 latency
    p95_latency = p99_latency = 0.0
    if len(latencies_ms) >= 20:
        qs = quantiles(latencies_ms, n=100)
        p95_latency = qs[94]  # 95th percentile (0-indexed)
        p99_latency = qs[98]  # 99th percentile
    elif latencies_ms:
        sorted_lat = sorted(latencies_ms)
        p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99_latency = sorted_lat[int(len(sorted_lat) * 0.99)]

    enemy_clear_rate = enemies_defeated_total / max(1, enemies_total)

    # ── Print results ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS ({n_games} games, {total_turns} total turns)")
    print(f"{'='*60}")
    print(f"\n  📊 Core Metrics")
    print(f"     Win rate:               {win_rate:.1%}  ({wins}/{n_games} games won)")
    print(f"     Schema compliance:      {schema_compliance:.1%}  ({total_turns - fallback_turns}/{total_turns} turns valid JSON)")
    print(f"     Enemy clear rate:       {enemy_clear_rate:.1%}  ({enemies_defeated_total}/{enemies_total} enemies defeated)")
    print(f"     Avg turns per game:     {avg_turns:.1f}")

    print(f"\n  ⚡ Latency (per turn, mock models)")
    print(f"     Avg:   {avg_latency:.2f} ms")
    print(f"     P95:   {p95_latency:.2f} ms")
    print(f"     P99:   {p99_latency:.2f} ms")

    print(f"\n  🎮 Action Distribution ({total_turns} total)")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / max(1, total_actions)
        print(f"     {action:<12} {count:>5}  ({pct:.1%})")

    print(f"\n  ⏱  Wall time: {wall_time:.1f}s for {n_games} games")
    print(f"{'='*60}\n")

    # ── Portfolio bullet suggestions ───────────────────────────────
    print("  📝 Portfolio-ready metrics:")
    print(f"     \"Win rate: {win_rate:.0%} across {n_games} offline games on 8x8 grid\"")
    print(f"     \"JSON schema compliance: {schema_compliance:.1%} ({total_turns - fallback_turns}/{total_turns} turns parsed without fallback)\"")
    print(f"     \"Avg turn latency: {avg_latency:.1f}ms (P99: {p99_latency:.1f}ms) with mock models\"")
    if enemy_clear_rate > 0:
        print(f"     \"Enemy clear rate: {enemy_clear_rate:.0%} across benchmark run\"")
    print()

    # ── Save JSON ──────────────────────────────────────────────────
    results = {
        "n_games": n_games,
        "total_turns": total_turns,
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "schema_compliance_rate": round(schema_compliance, 4),
        "fallback_turns": fallback_turns,
        "avg_turn_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "p99_latency_ms": round(p99_latency, 3),
        "avg_turns_per_game": round(avg_turns, 2),
        "enemy_clear_rate": round(enemy_clear_rate, 4),
        "action_distribution": action_counts,
        "wall_time_seconds": round(wall_time, 2),
    }

    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark game-companion-agent offline")
    parser.add_argument("--games", type=int, default=50, help="Number of games to run (default: 50)")
    parser.add_argument("--verbose", action="store_true", help="Print each turn")
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed offset")
    args = parser.parse_args()

    results = run_benchmark(
        n_games=args.games,
        verbose=args.verbose,
        seed_offset=args.seed_offset,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
