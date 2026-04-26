"""Game state renderer using PIL — cinematic HUD layout.

render(state)                              → grid-only image for VLM perception
render_composite(state, log, prev, score)  → full cinematic frame for GIF/viewer
make_intro_frame(state)                    → opening title card
"""

from __future__ import annotations

import random
from collections import deque
from PIL import Image, ImageDraw, ImageFont

from config import RenderConfig
from game.engine import GameState
from game.entities import ActionType

# ── Palette ────────────────────────────────────────────────────────────────────
_DARK       = (10, 10, 18)
_PANEL_BG   = (14, 14, 24)
_CELL_BG    = (22, 22, 36)
_GRID_LINE  = (34, 34, 54)
_CYAN       = (0, 215, 230)
_CYAN_DIM   = (0, 125, 140)
_WHITE      = (245, 245, 245)
_OFFWHITE   = (190, 190, 208)
_GRAY       = (105, 105, 130)
_DIVIDER    = (38, 38, 58)
_GREEN      = (45, 210, 75)
_RED        = (230, 48, 48)
_ORANGE     = (255, 160, 30)
_BLUE       = (50, 135, 250)
_GOLD       = (250, 205, 45)
_PURPLE     = (160, 80, 255)
_PATH_DOT   = (0, 180, 200)
_HIT_FLASH  = (255, 80, 30)
_HEAL_FLASH = (50, 240, 100)
_DMG_COLOR  = (255, 90, 40)
_KILL_COLOR = (255, 220, 50)
_SCORE_COLOR= (0, 215, 230)

_LABEL_COLS = "ABCDEFGH"


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _blend(col: tuple, factor: float) -> tuple:
    return tuple(_clamp(int(c * factor)) for c in col)


def _lighten(col: tuple, factor: float) -> tuple:
    return tuple(_clamp(int(c + (255 - c) * factor)) for c in col)


class GameRenderer:
    """Renders game state to PIL Images with professional HUD."""

    LABEL_MARGIN = 22
    PANEL_W      = 275
    GUTTER       = 8

    def __init__(self, config: RenderConfig) -> None:
        self.cfg  = config
        self.cell = config.cell_size
        self.pad  = config.padding

        self._fsm  = self._load_font(10)
        self._fmd  = self._load_font(12)
        self._flg  = self._load_font(15)
        self._fxl  = self._load_font(21)
        self._fxxl = self._load_font(30)

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, state: GameState) -> Image.Image:
        """Grid-only image — unchanged API for VLM perception."""
        return self._make_grid_image(state, None, None, False, [], None)

    def render_composite(self,
                         state: GameState,
                         log=None,
                         prev_state: GameState | None = None,
                         score: int = 0) -> Image.Image:
        """Full cinematic frame: labelled grid + HUD panel."""
        highlight, path, floating_texts, shake, kill_banner = \
            self._compute_overlays(state, log, prev_state)

        grid_img = self._make_grid_image(
            state, highlight, path, shake, floating_texts, kill_banner)
        labelled  = self._add_axis_labels(grid_img, state)
        panel_img = self._make_panel(state, log, score, target_h=labelled.height)

        total_w = labelled.width + self.GUTTER + panel_img.width
        canvas  = Image.new("RGB", (total_w, labelled.height), _DARK)
        canvas.paste(labelled,  (0, 0))
        canvas.paste(panel_img, (labelled.width + self.GUTTER, 0))
        return canvas

    def make_intro_frame(self, state: GameState) -> Image.Image:
        """Opening title card shown before turn 1."""
        dummy = self.render_composite(state, log=None, prev_state=None, score=0)
        w, h  = dummy.size

        img  = Image.new("RGB", (w, h), _DARK)
        draw = ImageDraw.Draw(img)

        for gx in range(state.grid_width):
            for gy in range(state.grid_height):
                xo = self.LABEL_MARGIN + self.pad + gx * self.cell
                yo = self.LABEL_MARGIN + self.pad + gy * self.cell
                draw.rectangle([xo, yo, xo + self.cell - 1, yo + self.cell - 1],
                               fill=(16, 16, 26), outline=(28, 28, 44))

        gw  = state.grid_width  * self.cell
        gh  = state.grid_height * self.cell
        gx0 = self.LABEL_MARGIN + self.pad
        gy0 = self.LABEL_MARGIN + self.pad
        self._brackets(draw, gx0, gy0, gx0 + gw, gy0 + gh, _CYAN)

        cx, cy = (gx0 + gw) // 2, h // 2

        title1 = "GRIDWORLD ARENA"
        title2 = "AI Companion Agent"
        sub    = f"{state.grid_width}×{state.grid_height} grid · {state.grid_width * state.grid_height} cells"

        b1 = draw.textbbox((0, 0), title1, font=self._fxxl)
        b2 = draw.textbbox((0, 0), title2, font=self._flg)
        bs = draw.textbbox((0, 0), sub,    font=self._fsm)

        for offset in (20, 14, 8, 4):
            draw.text((cx - (b1[2]-b1[0])//2 - offset//2,
                       cy - 40 - offset//2),
                      title1, fill=_blend(_CYAN, 0.05), font=self._fxxl)

        draw.text((cx - (b1[2]-b1[0])//2, cy - 40), title1,
                  fill=_CYAN, font=self._fxxl)
        draw.text((cx - (b2[2]-b2[0])//2, cy - 40 + (b1[3]-b1[1]) + 4),
                  title2, fill=_OFFWHITE, font=self._flg)
        draw.text((cx - (bs[2]-bs[0])//2, cy + 18), sub,
                  fill=_GRAY, font=self._fsm)

        legend = [("P", _BLUE, "Player"), ("E", _RED, "Enemy"),
                  ("+", _GREEN, "Health Pack"), ("G", _GOLD, "Goal")]
        lx = cx - 150
        ly = cy + 48
        for _, col, label in legend:
            r = 8
            draw.ellipse([lx, ly, lx + r*2, ly + r*2], fill=col)
            draw.text((lx + r*2 + 5, ly), label, fill=_OFFWHITE, font=self._fsm)
            lx += 80

        px = gx0 + gw + self.GUTTER
        panel = self._make_panel(state, log=None, score=0, target_h=h)
        img.paste(panel, (px, 0))
        return img

    # ── Grid image ─────────────────────────────────────────────────────────────

    def _make_grid_image(self,
                         state: GameState,
                         highlight_cell,
                         path,
                         shake: bool,
                         floating_texts: list,
                         kill_banner) -> Image.Image:
        w = state.grid_width  * self.cell + self.pad * 2
        h = state.grid_height * self.cell + self.pad * 2

        # Screen shake: render on slightly larger canvas then crop
        if shake:
            sx = random.randint(-4, 4)
            sy = random.randint(-3, 3)
        else:
            sx, sy = 0, 0

        img  = Image.new("RGB", (w, h), _DARK)
        draw = ImageDraw.Draw(img)

        for gx in range(state.grid_width):
            for gy in range(state.grid_height):
                x0 = self.pad + gx * self.cell + sx
                y0 = self.pad + gy * self.cell + sy
                x1 = x0 + self.cell - 1
                y1 = y0 + self.cell - 1
                draw.rectangle([x0, y0, x1, y1], fill=_CELL_BG, outline=_GRID_LINE)

        if path:
            self._draw_path(img, draw, path, sx, sy)

        if highlight_cell:
            hx, hy, hcol = highlight_cell
            self._draw_cell_highlight(draw, hx, hy, hcol, sx, sy)

        self._draw_entity(img, draw, state.goal.x, state.goal.y, _GOLD, "G", sx, sy)
        for hp in state.available_health_packs:
            self._draw_entity(img, draw, hp.x, hp.y, _GREEN, "+", sx, sy)
        for enemy in state.living_enemies:
            self._draw_entity(img, draw, enemy.x, enemy.y,
                              _RED, f"E{enemy.entity_id}", sx, sy)
            self._draw_hp_bar(draw, enemy.x, enemy.y,
                              enemy.hp, enemy.max_hp, _RED, sx, sy)
        self._draw_entity(img, draw, state.player.x, state.player.y,
                          _BLUE, "P", sx, sy)
        self._draw_hp_bar(draw, state.player.x, state.player.y,
                          state.player.hp, state.player.max_hp, _BLUE, sx, sy)

        # Floating damage / heal texts
        for (fgx, fgy, text, color) in floating_texts:
            self._draw_floating_text(draw, fgx, fgy, text, color, sx, sy)

        # Kill banner
        if kill_banner:
            self._draw_kill_banner(img, kill_banner, w, h)

        # Scanlines
        self._apply_scanlines(img)

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w - 1, h - 1], outline=_CYAN_DIM, width=2)
        self._brackets(draw, 0, 0, w - 1, h - 1, _CYAN, length=14, width=2)

        if state.game_over:
            self._draw_game_over_overlay(img, state)

        return img

    # ── Axis labels ─────────────────────────────────────────────────────────────

    def _add_axis_labels(self, grid_img: Image.Image,
                         state: GameState) -> Image.Image:
        m    = self.LABEL_MARGIN
        w    = grid_img.width  + m
        h    = grid_img.height + m
        out  = Image.new("RGB", (w, h), _DARK)
        draw = ImageDraw.Draw(out)
        out.paste(grid_img, (m, m))

        for gx in range(state.grid_width):
            cx     = m + self.pad + gx * self.cell + self.cell // 2
            letter = _LABEL_COLS[gx] if gx < len(_LABEL_COLS) else str(gx)
            bb     = draw.textbbox((0, 0), letter, font=self._fsm)
            draw.text((cx - (bb[2]-bb[0])//2, 4),
                      letter, fill=_CYAN_DIM, font=self._fsm)

        for gy in range(state.grid_height):
            cy  = m + self.pad + gy * self.cell + self.cell // 2
            num = str(gy + 1)
            bb  = draw.textbbox((0, 0), num, font=self._fsm)
            draw.text((m - bb[2] - 3, cy - (bb[3]-bb[1])//2),
                      num, fill=_CYAN_DIM, font=self._fsm)
        return out

    # ── Entity rendering ────────────────────────────────────────────────────────

    def _draw_entity(self, img, draw, gx, gy, color, label,
                     sx=0, sy=0) -> None:
        cx = self.pad + gx * self.cell + self.cell // 2 + sx
        cy = self.pad + gy * self.cell + self.cell // 2 + sy
        r  = self.cell // 3

        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd   = ImageDraw.Draw(glow)
        for offset, alpha in ((r + 12, 20), (r + 7, 45), (r + 3, 75)):
            gd.ellipse([cx-offset, cy-offset, cx+offset, cy+offset],
                       fill=(*color, alpha))
        img.paste(Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB"))

        draw = ImageDraw.Draw(img)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
        hi = _lighten(color, 0.45)
        draw.ellipse([cx-r+3, cy-r+3, cx-r+9, cy-r+9], fill=hi)

        font = self._fmd
        bb   = draw.textbbox((0, 0), label, font=font)
        tw   = bb[2] - bb[0]
        th   = bb[3] - bb[1]
        draw.text((cx - tw//2 + 1, cy - th//2 + 1), label, fill=(0,0,0), font=font)
        draw.text((cx - tw//2,     cy - th//2 - 1), label, fill=_WHITE,  font=font)

    def _draw_hp_bar(self, draw, gx, gy, hp, max_hp, color,
                     sx=0, sy=0) -> None:
        bw = self.cell - 8
        x0 = self.pad + gx * self.cell + 4 + sx
        y0 = self.pad + gy * self.cell + self.cell - 10 + sy
        draw.rectangle([x0, y0, x0 + bw, y0 + 5], fill=(28, 28, 42))
        fw = int(bw * (hp / max_hp))
        if fw > 0:
            draw.rectangle([x0, y0, x0 + fw, y0 + 5], fill=color)
            draw.rectangle([x0, y0, x0 + fw, y0 + 1], fill=_lighten(color, 0.5))

    def _draw_cell_highlight(self, draw, gx, gy, color, sx=0, sy=0) -> None:
        x0 = self.pad + gx * self.cell + 1 + sx
        y0 = self.pad + gy * self.cell + 1 + sy
        x1 = x0 + self.cell - 3
        y1 = y0 + self.cell - 3
        for i, af in enumerate((0.25, 0.45, 0.70)):
            c = tuple(_clamp(int(c * af + 30)) for c in color)
            draw.rectangle([x0+i, y0+i, x1-i, y1-i], outline=c, width=1)

    def _draw_path(self, img, draw, path, sx=0, sy=0) -> None:
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd   = ImageDraw.Draw(glow)
        for i, (px, py) in enumerate(path):
            cx = self.pad + px * self.cell + self.cell // 2 + sx
            cy = self.pad + py * self.cell + self.cell // 2 + sy
            alpha = max(40, 120 - i * 10)
            gd.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(*_PATH_DOT, alpha))
            draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=_blend(_PATH_DOT, 0.6))
        img.paste(Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB"))

    # ── NEW: Floating damage / heal text ───────────────────────────────────────

    def _draw_floating_text(self, draw, gx: int, gy: int,
                            text: str, color: tuple,
                            sx: int = 0, sy: int = 0) -> None:
        """Draw bold floating text above a grid cell (damage / heal numbers)."""
        cx = self.pad + gx * self.cell + self.cell // 2 + sx
        # Float 18px above the entity center
        cy = self.pad + gy * self.cell + self.cell // 2 - 18 + sy

        font = self._flg
        bb   = draw.textbbox((0, 0), text, font=font)
        tw   = bb[2] - bb[0]
        th   = bb[3] - bb[1]

        # Shadow
        draw.text((cx - tw//2 + 2, cy - th//2 + 2), text,
                  fill=(0, 0, 0), font=font)
        # Main text
        draw.text((cx - tw//2, cy - th//2), text, fill=color, font=font)

    # ── NEW: Kill / event banner ────────────────────────────────────────────────

    def _draw_kill_banner(self, img: Image.Image,
                          text: str, w: int, h: int) -> None:
        """Semi-transparent banner across the grid with event text."""
        banner_h = 28
        by0      = 10
        by1      = by0 + banner_h

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od      = ImageDraw.Draw(overlay)
        od.rectangle([4, by0, w - 4, by1], fill=(0, 0, 0, 180))
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

        draw = ImageDraw.Draw(img)
        font = self._fmd
        bb   = draw.textbbox((0, 0), text, font=font)
        tw   = bb[2] - bb[0]
        th   = bb[3] - bb[1]
        cx   = w // 2
        cy   = by0 + banner_h // 2

        draw.text((cx - tw//2 + 1, cy - th//2 + 1), text,
                  fill=(0, 0, 0), font=font)
        draw.text((cx - tw//2, cy - th//2), text, fill=_KILL_COLOR, font=font)

    # ── Other visual helpers ────────────────────────────────────────────────────

    def _apply_scanlines(self, img: Image.Image) -> None:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od      = ImageDraw.Draw(overlay)
        for y in range(0, img.height, 4):
            od.line([(0, y), (img.width, y)], fill=(0, 0, 0, 28))
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

    def _draw_game_over_overlay(self, img: Image.Image,
                                state: GameState) -> None:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 150))
        img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))

        draw = ImageDraw.Draw(img)
        text, color = (("VICTORY!", _GOLD) if state.won
                       else ("DEFEATED", _RED) if state.player.is_dead
                       else ("TIME OUT", _GRAY))
        font = self._fxl
        bb   = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        cx, cy = img.width//2, img.height//2
        draw.text((cx - tw//2 + 2, cy - th//2 + 2), text, fill=(0,0,0), font=font)
        draw.text((cx - tw//2,     cy - th//2),     text, fill=color,   font=font)

    @staticmethod
    def _brackets(draw, x0, y0, x1, y1, color,
                  length=12, width=2) -> None:
        for bx, by, dx, dy in (
            (x0, y0,  1,  1), (x1, y0, -1,  1),
            (x0, y1,  1, -1), (x1, y1, -1, -1),
        ):
            draw.line([(bx, by), (bx + dx*length, by)], fill=color, width=width)
            draw.line([(bx, by), (bx, by + dy*length)], fill=color, width=width)

    # ── Overlay computation ─────────────────────────────────────────────────────

    def _compute_overlays(self, state, log, prev_state):
        """Return (highlight, path, floating_texts, shake, kill_banner)."""
        highlight      = None
        path           = None
        floating_texts = []
        shake          = False
        kill_banner    = None

        if log is not None and log.chosen_action is not None:
            at = log.chosen_action.action_type

            if at == ActionType.ATTACK and log.chosen_action.direction:
                dx, dy = log.chosen_action.direction.to_delta()
                if prev_state:
                    tx = prev_state.player.x + dx
                    ty = prev_state.player.y + dy
                else:
                    tx = state.player.x + dx
                    ty = state.player.y + dy

                if 0 <= tx < state.grid_width and 0 <= ty < state.grid_height:
                    highlight = (tx, ty, _HIT_FLASH)
                    floating_texts.append((tx, ty, "−25", _DMG_COLOR))
                    shake = True

            elif at == ActionType.MOVE:
                highlight = (state.player.x, state.player.y, _BLUE)

            elif at == ActionType.HEAL:
                highlight = (state.player.x, state.player.y, _HEAL_FLASH)
                floating_texts.append(
                    (state.player.x, state.player.y, "+30 HP", _GREEN))

        # Kill detection
        if prev_state is not None:
            prev_alive = len(prev_state.living_enemies)
            curr_alive = len(state.living_enemies)
            killed     = prev_alive - curr_alive
            if killed == 1:
                kill_banner = "  ☆  ENEMY DOWN!  ☆  "
            elif killed > 1:
                kill_banner = f"  ☆  {killed}x MULTI-KILL!  ☆  "

        if not state.game_over:
            path = self._bfs_path(state)

        return highlight, path, floating_texts, shake, kill_banner

    def _bfs_path(self, state) -> list:
        blocked = {(e.x, e.y) for e in state.living_enemies}
        start   = (state.player.x, state.player.y)
        goal    = (state.goal.x,   state.goal.y)
        if start == goal:
            return []
        queue   = deque([(start, [])])
        visited = {start}
        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy in ((0,-1),(0,1),(1,0),(-1,0)):
                nx, ny = cx+dx, cy+dy
                npos   = (nx, ny)
                if npos == goal:
                    return path + [npos]
                if (0 <= nx < state.grid_width
                        and 0 <= ny < state.grid_height
                        and npos not in visited
                        and npos not in blocked):
                    visited.add(npos)
                    queue.append((npos, path + [npos]))
        return []

    # ── HUD Panel ───────────────────────────────────────────────────────────────

    def _make_panel(self, state, log, score: int, target_h: int) -> Image.Image:
        img  = Image.new("RGB", (self.PANEL_W, target_h), _PANEL_BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, self.PANEL_W-1, target_h-1],
                       outline=_CYAN_DIM, width=1)
        self._brackets(draw, 0, 0, self.PANEL_W-1, target_h-1,
                       _CYAN, length=10, width=1)

        y  = 12
        pw = self.PANEL_W

        draw.text((12, y), "AI COMPANION", fill=_CYAN,    font=self._flg)
        y += 20
        draw.text((12, y), "GRIDWORLD ARENA", fill=_CYAN_DIM, font=self._fsm)
        y += 16
        self._hline(draw, y, pw)
        y += 10

        def stat(label, value, vc=_OFFWHITE):
            nonlocal y
            draw.text((12, y),  label, fill=_GRAY, font=self._fsm)
            draw.text((110, y), str(value), fill=vc, font=self._fsm)
            y += 16

        stat("TURN",      str(state.turn))

        # Score with cyan highlight
        if score > 0:
            stat("SCORE", f"{score:,}", _SCORE_COLOR)

        hp_pct   = state.player.hp / state.player.max_hp
        hp_color = (_GREEN  if hp_pct > 0.5
                    else _ORANGE if hp_pct > 0.25
                    else _RED)
        stat("PLAYER HP", f"{state.player.hp}/{state.player.max_hp}", hp_color)
        stat("ENEMIES",   str(len(state.living_enemies)),
             _RED if state.living_enemies else _GREEN)
        stat("GOAL DIST", str(int(state.player.distance_to(state.goal))))

        if log is not None:
            if log.chosen_action:
                act = log.chosen_action.action_type.name
                if log.chosen_action.direction:
                    act += f" {log.chosen_action.direction.name}"
                stat("ACTION", act, _CYAN)
            if log.risk_level:
                rl = log.risk_level.upper()
                rc = (_RED if rl in ("HIGH","CRITICAL")
                      else _ORANGE if rl == "MEDIUM" else _GREEN)
                stat("RISK", rl, rc)
            if log.persona:
                stat("PERSONA", log.persona.upper(), _PURPLE)

        y += 4
        self._hline(draw, y, pw); y += 10

        draw.text((12, y), "COMPANION SAYS", fill=_CYAN, font=self._fsm)
        y += 16
        msg = (log.companion_message
               if log is not None and log.companion_message
               else "Analysing situation...")
        for line in self._wrap(msg, 31):
            draw.text((14, y), line, fill=_OFFWHITE, font=self._fsm)
            y += 13
        y += 6
        self._hline(draw, y, pw); y += 10

        draw.text((12, y), "PLAN", fill=_CYAN, font=self._fsm)
        y += 16
        plan = log.plan[:3] if (log is not None and log.plan) else []
        if plan:
            for i, step in enumerate(plan, 1):
                for j, line in enumerate(self._wrap(step, 29)):
                    draw.text((14, y),
                              (f"{i}. " if j == 0 else "   ") + line,
                              fill=_OFFWHITE, font=self._fsm)
                    y += 13
        else:
            draw.text((14, y), "—", fill=_GRAY, font=self._fsm); y += 14
        y += 6
        self._hline(draw, y, pw); y += 10

        draw.text((12, y), "LEGEND", fill=_CYAN, font=self._fsm)
        y += 16
        for col, label in ((_BLUE,"P  Player"), (_RED,"E  Enemy"),
                            (_GREEN,"+ Health Pack"), (_GOLD,"G  Goal")):
            r = 6
            draw.ellipse([13, y+1, 13+r*2, y+1+r*2], fill=col)
            draw.text((13+r*2+7, y), label, fill=_OFFWHITE, font=self._fsm)
            y += 18

        if state.game_over:
            banner_y = max(y + 14, target_h - 46)
            self._hline(draw, banner_y, pw)
            text, color = (("VICTORY!", _GOLD) if state.won
                           else ("DEFEATED", _RED) if state.player.is_dead
                           else ("TIME OUT", _GRAY))
            bb  = draw.textbbox((0, 0), text, font=self._fxl)
            tw  = bb[2] - bb[0]
            tx  = (pw - tw) // 2
            ty  = banner_y + 8
            draw.text((tx+2, ty+2), text, fill=(0,0,0),  font=self._fxl)
            draw.text((tx,   ty),   text, fill=color,    font=self._fxl)

        return img

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _hline(self, draw, y, pw):
        draw.line([(12, y), (pw-12, y)], fill=_DIVIDER, width=1)

    @staticmethod
    def _wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= width:
                cur = (cur + " " + w).strip()
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines or [""]

    @staticmethod
    def _load_font(size):
        candidates = [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
