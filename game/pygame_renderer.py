"""Pygame visual renderer for GridWorld Arena.

Animated real-time display with glow effects, HP bars, and an
AI companion HUD panel. Replays a pre-recorded game session so
no GPU is required — just run run_visual.py.

Controls during replay:
  SPACE / RIGHT ARROW  — skip to next turn
  LEFT ARROW           — step back one turn
  S                    — save screenshot
  Q / ESC              — quit
"""

import math
import sys
from typing import Optional

import pygame

from game.engine import GameState

# ── Color palette (mirrors portfolio: #06060f, violet, cyan, green) ──────────
BG        = (  6,   6,  15)
GRID_BG   = ( 12,  12,  24)
GRID_LINE = ( 28,  28,  46)
PLAYER_C  = ( 50, 130, 240)
ENEMY_C   = (220,  50,  50)
HP_C      = ( 50, 200,  80)
GOAL_C    = (240, 200,  50)
HUD_BG    = (  8,   8,  20)
WHITE     = (220, 220, 230)
GRAY      = (110, 110, 135)
DIM       = ( 45,  45,  65)
VIOLET    = (124,  92, 252)
CYAN      = (  6, 182, 212)
GREEN     = ( 52, 211, 153)
HP_BAR_BG = ( 35,  35,  52)

# ── Layout constants ──────────────────────────────────────────────────────────
CELL   = 74      # px per grid cell
PAD    = 14      # grid padding
HUD_W  = 350     # HUD panel width
FPS    = 60
TURN_MS = 480    # ms between auto-advance turns


class PygameRenderer:
    """Animated Pygame renderer for GridWorld Arena.

    Usage:
        renderer = PygameRenderer(grid_w=8, grid_h=8)
        renderer.replay(states, logs)   # blocks until window closed
    """

    def __init__(self, grid_w: int = 8, grid_h: int = 8) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h

        self.win_w = PAD * 2 + grid_w * CELL + HUD_W
        self.win_h = PAD * 2 + grid_h * CELL

        pygame.init()
        pygame.display.set_caption("GridWorld Arena  —  AI Companion Agent")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

        # Fonts
        pygame.font.init()
        mono = pygame.font.match_font("jetbrainsmono,couriernew,monospace")
        sans = pygame.font.match_font("inter,sora,segoeui,helvetica,arial")
        self.f_sm  = pygame.font.Font(mono,  12)
        self.f_md  = pygame.font.Font(mono,  14)
        self.f_lg  = pygame.font.Font(sans,  18)
        self.f_xl  = pygame.font.Font(sans,  22)
        self.f_hud = pygame.font.Font(mono,  13)

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def _cell_rect(self, gx: int, gy: int) -> pygame.Rect:
        return pygame.Rect(PAD + gx * CELL, PAD + gy * CELL, CELL, CELL)

    # ── Drawing primitives ───────────────────────────────────────────────────

    def _draw_grid(self) -> None:
        area = pygame.Rect(PAD, PAD, self.grid_w * CELL, self.grid_h * CELL)
        pygame.draw.rect(self.screen, GRID_BG, area)
        for gx in range(self.grid_w + 1):
            x = PAD + gx * CELL
            pygame.draw.line(self.screen, GRID_LINE,
                             (x, PAD), (x, PAD + self.grid_h * CELL))
        for gy in range(self.grid_h + 1):
            y = PAD + gy * CELL
            pygame.draw.line(self.screen, GRID_LINE,
                             (PAD, y), (PAD + self.grid_w * CELL, y))

    def _glow_circle(self, cx: int, cy: int, r: int,
                     color: tuple, intensity: float = 1.0) -> None:
        """Draw a glowing circle using layered alpha surfaces."""
        for extra, alpha in ((16, 12), (11, 22), (6, 38), (3, 55)):
            lr = r + extra
            surf = pygame.Surface((lr * 2 + 2, lr * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, int(alpha * intensity)),
                               (lr + 1, lr + 1), lr)
            self.screen.blit(surf, (cx - lr - 1, cy - lr - 1))
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        # Specular highlight
        hi = tuple(min(255, v + 70) for v in color)
        pygame.draw.circle(self.screen, hi,
                           (cx - r // 3, cy - r // 3), max(2, r // 3))

    def _label(self, cx: int, cy: int, text: str,
               color: tuple = WHITE, font: Optional[pygame.font.Font] = None) -> None:
        f = font or self.f_sm
        s = f.render(text, True, color)
        self.screen.blit(s, s.get_rect(center=(cx, cy)))

    def _hp_bar(self, gx: int, gy: int, hp: int, max_hp: int,
                color: tuple) -> None:
        r = self._cell_rect(gx, gy)
        bx, by = r.x + 6, r.bottom - 10
        bw, bh = CELL - 12, 5
        pygame.draw.rect(self.screen, HP_BAR_BG, (bx, by, bw, bh))
        fill = int(bw * max(0, hp) / max_hp)
        if fill > 0:
            pygame.draw.rect(self.screen, color, (bx, by, fill, bh))

    # ── Entity renderers ─────────────────────────────────────────────────────

    def _draw_player(self, gx: int, gy: int, hp: int, max_hp: int) -> None:
        r = self._cell_rect(gx, gy)
        cx, cy = r.centerx, r.centery
        self._glow_circle(cx, cy, CELL // 3 - 1, PLAYER_C)
        self._label(cx, cy, "P")
        self._hp_bar(gx, gy, hp, max_hp, PLAYER_C)

    def _draw_enemy(self, gx: int, gy: int, hp: int, max_hp: int,
                    eid: int, t: int) -> None:
        r = self._cell_rect(gx, gy)
        cx, cy = r.centerx, r.centery
        pulse = 0.75 + 0.25 * math.sin(t / 380.0 + eid * 1.5)
        self._glow_circle(cx, cy, CELL // 3 - 1, ENEMY_C, intensity=pulse)
        self._label(cx, cy, f"E{eid}")
        self._hp_bar(gx, gy, hp, max_hp, ENEMY_C)

    def _draw_health_pack(self, gx: int, gy: int) -> None:
        r = self._cell_rect(gx, gy)
        cx, cy = r.centerx, r.centery
        arm = CELL // 4
        thick = 5
        # Glow
        surf = pygame.Surface((arm * 4, arm * 4), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*HP_C, 35), (arm * 2, arm * 2), arm * 2)
        self.screen.blit(surf, (cx - arm * 2, cy - arm * 2))
        # Cross
        pygame.draw.rect(self.screen, HP_C,
                         (cx - thick // 2, cy - arm, thick, arm * 2))
        pygame.draw.rect(self.screen, HP_C,
                         (cx - arm, cy - thick // 2, arm * 2, thick))

    def _draw_goal(self, gx: int, gy: int, t: int) -> None:
        r = self._cell_rect(gx, gy)
        cx, cy = r.centerx, r.centery
        pulse = 0.5 + 0.5 * math.sin(t / 280.0)
        radius = int((CELL // 4) * (0.85 + 0.15 * pulse))
        self._glow_circle(cx, cy, radius, GOAL_C, intensity=0.5 + 0.5 * pulse)
        self._label(cx, cy, "★", GOAL_C)

    def _draw_cell_flash(self, gx: int, gy: int, t: int,
                         end_ms: int, color: tuple) -> None:
        remaining = end_ms - t
        if remaining <= 0:
            return
        alpha = int(160 * (remaining / 320.0))
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        surf.fill((*color, alpha))
        self.screen.blit(surf, self._cell_rect(gx, gy).topleft)

    # ── HUD panel ────────────────────────────────────────────────────────────

    def _wrap(self, text: str, max_chars: int) -> list[str]:
        if not text:
            return []
        words = text.split()
        lines, buf = [], []
        for w in words:
            if sum(len(x) + 1 for x in buf) + len(w) <= max_chars:
                buf.append(w)
            else:
                if buf:
                    lines.append(" ".join(buf))
                buf = [w]
        if buf:
            lines.append(" ".join(buf))
        return lines

    def _draw_hud(self, state: GameState, log, t: int) -> None:  # log: TurnLog | None
        hx = PAD * 2 + self.grid_w * CELL
        pygame.draw.rect(self.screen, HUD_BG,
                         (hx, 0, self.win_w - hx, self.win_h))
        pygame.draw.line(self.screen, VIOLET, (hx, 0), (hx, self.win_h), 1)

        y = 16
        x = hx + 14
        rw = self.win_w - hx - 28   # usable text width in px

        def hdr(text: str, color: tuple = VIOLET) -> None:
            nonlocal y
            s = self.f_lg.render(text, True, color)
            self.screen.blit(s, (x, y))
            y += s.get_height() + 3

        def row(label: str, val: str,
                lc: tuple = GRAY, vc: tuple = WHITE) -> None:
            nonlocal y
            ls = self.f_hud.render(f"{label}:", True, lc)
            vs = self.f_hud.render(val, True, vc)
            self.screen.blit(ls, (x, y))
            self.screen.blit(vs, (x + ls.get_width() + 6, y))
            y += ls.get_height() + 3

        def div() -> None:
            nonlocal y
            y += 5
            pygame.draw.line(self.screen, GRID_LINE, (x, y), (x + rw, y))
            y += 9

        def block(lines: list[str], color: tuple = WHITE) -> None:
            nonlocal y
            for line in lines:
                s = self.f_hud.render(line, True, color)
                self.screen.blit(s, (x, y))
                y += s.get_height() + 2
            y += 3

        # Title
        hdr("AI COMPANION", VIOLET)
        hdr("GRIDWORLD ARENA", CYAN)
        div()

        # Stats
        row("TURN", str(state.turn), GRAY, CYAN)
        hp, mhp = state.player.hp, state.player.max_hp
        hp_col = GREEN if hp > mhp * 0.6 else (GOAL_C if hp > mhp * 0.3 else ENEMY_C)
        row("PLAYER HP", f"{hp}/{mhp}", GRAY, hp_col)
        row("ENEMIES",   f"{len(state.living_enemies)} alive", GRAY, WHITE)
        row("GOAL DIST", str(state.player.distance_to(state.goal)), GRAY, WHITE)
        div()

        if log is not None:
            act_str = str(log.chosen_action).upper() if log.chosen_action else "—"
            act_col = CYAN if not log.action_was_fallback else GOAL_C
            row("ACTION", act_str, GRAY, act_col)
            if log.action_was_fallback:
                row("MODE", "heuristic", GRAY, GOAL_C)
            if log.risk_level:
                rc = {"low": GREEN, "medium": GOAL_C, "high": ENEMY_C}.get(
                    log.risk_level.lower(), WHITE)
                row("RISK", log.risk_level.upper(), GRAY, rc)
            div()

            if log.companion_message:
                hdr("COMPANION SAYS", CYAN)
                block(self._wrap(log.companion_message, 33)[:4], WHITE)
                div()

            if log.plan:
                hdr("PLAN", GRAY)
                for i, step in enumerate(log.plan[:3]):
                    block(self._wrap(f"{i+1}. {step}", 33)[:2], DIM)

        div()

        # Legend
        hdr("LEGEND", GRAY)
        for color, label in [
            (PLAYER_C, "P  — Player"),
            (ENEMY_C,  "E# — Enemy"),
            (HP_C,     "+  — Health Pack"),
            (GOAL_C,   "★  — Goal (reach to win)"),
        ]:
            s = self.f_hud.render(label, True, color)
            self.screen.blit(s, (x, y))
            y += s.get_height() + 3

        # Bottom bar
        if state.game_over:
            msg = "VICTORY!  Goal reached." if state.won else "DEFEAT.  Agent eliminated."
            col = GREEN if state.won else ENEMY_C
            s = self.f_lg.render(msg, True, col)
            self.screen.blit(s, (x, self.win_h - 42))
        else:
            pulse = 0.5 + 0.5 * math.sin(t / 500.0)
            col = tuple(int(v * pulse) for v in GREEN)
            s = self.f_hud.render("● REPLAYING  (SPACE=next  ESC=quit)", True, col)
            self.screen.blit(s, (x, self.win_h - 28))

    # ── Public API ───────────────────────────────────────────────────────────

    def render_frame(self, state: GameState, log=None,
                     flashes: Optional[list] = None) -> None:
        """Draw one frame. Call pygame.display.flip() automatically."""
        t = pygame.time.get_ticks()
        self.screen.fill(BG)
        self._draw_grid()

        if flashes:
            for gx, gy, color, end_ms in flashes:
                self._draw_cell_flash(gx, gy, t, end_ms, color)

        self._draw_goal(state.goal.x, state.goal.y, t)

        for hp_item in state.available_health_packs:
            self._draw_health_pack(hp_item.x, hp_item.y)

        for enemy in state.living_enemies:
            self._draw_enemy(enemy.x, enemy.y,
                             enemy.hp, enemy.max_hp, enemy.entity_id, t)

        self._draw_player(state.player.x, state.player.y,
                          state.player.hp, state.player.max_hp)

        self._draw_hud(state, log, t)
        pygame.display.flip()

    def replay(self, states: list, logs: list,
               turn_ms: int = TURN_MS,
               screenshot_path: str = "gameplay_preview.png") -> None:
        """Replay a full recorded game.

        Args:
            states: List of GameState snapshots (one per turn, starting from turn 0).
            logs:   List of TurnLog objects aligned to states (None for the initial state).
            turn_ms: Milliseconds between auto-advance turns.
            screenshot_path: Where to auto-save a mid-game screenshot.
        """
        if not states:
            return

        from game.entities import ActionType

        flashes: list = []
        idx = 0
        last_advance = pygame.time.get_ticks()
        screenshot_saved = False
        mid_idx = len(states) // 2  # frame to auto-save

        while True:
            t = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        return
                    if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                        if idx < len(states) - 1:
                            idx += 1
                            last_advance = t
                    if event.key == pygame.K_LEFT:
                        if idx > 0:
                            idx -= 1
                            last_advance = t
                    if event.key == pygame.K_s:
                        pygame.image.save(self.screen, screenshot_path)
                        print(f"Screenshot saved: {screenshot_path}")

            # Auto-advance
            if t - last_advance > turn_ms and idx < len(states) - 1:
                idx += 1
                last_advance = t
                log = logs[idx] if idx < len(logs) else None
                if log and log.chosen_action:
                    if log.chosen_action.action_type == ActionType.ATTACK:
                        state = states[idx]
                        # Flash dead enemies
                        for enemy in state.enemies:
                            if not enemy.alive:
                                flashes.append((enemy.x, enemy.y, ENEMY_C, t + 320))

            # Auto-save screenshot at mid-game
            if not screenshot_saved and idx >= mid_idx and mid_idx > 0:
                self.render_frame(states[idx],
                                  logs[idx] if idx < len(logs) else None, flashes)
                pygame.image.save(self.screen, screenshot_path)
                print(f"Auto-saved screenshot: {screenshot_path}")
                screenshot_saved = True

            flashes = [(gx, gy, c, em) for gx, gy, c, em in flashes if t < em]

            self.render_frame(states[idx],
                              logs[idx] if idx < len(logs) else None, flashes)
            self.clock.tick(FPS)

    def close(self) -> None:
        pygame.quit()
