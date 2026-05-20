"""ppcf - Superficie Pitch Control (Spearman 2018) sobre el campo.

Layout pitch-aligned (header + pitch + footer en el MISMO ancho horizontal)
portado de jaime-oriol/Diagonality_3D (viz/passes_plot.py): header con
escudo de la seleccion atacante + titulo auto + logo JO; footer con 3
bloques (leyenda colormap PPCF, direccion de ataque, stats del frame).

Uso:
    python -m src.viz.ppcf <match_id> <period> <minute>
    python -m src.viz.ppcf                          # default Messi ARG-MEX
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as mpatches
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import Z02_pitch_control as pc
from M01_loader_pff import scan_tracking, load_metadata, load_rosters

from viz.common import (ATT, BALL, BG, DEF, GK, PE_S, PITCH_LENGTH,
                         PITCH_WIDTH, PPCF_CMAP, WHITE, draw_pitch, _LOGO_PATH)

_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"

# pff team_name -> iso3 slug (sportlogos/sport.db.logos) — necesario para
# escoger el escudo correcto a partir del team_id atacante.
_TEAM_TO_SLUG = {
    "Argentina":"arg","Brazil":"bra","Ecuador":"ecu","Uruguay":"uru","Belgium":"bel",
    "Croatia":"cro","Denmark":"den","England":"eng","France":"fra","Germany":"ger",
    "Netherlands":"ned","Poland":"pol","Portugal":"por","Serbia":"srb","Spain":"esp",
    "Switzerland":"sui","Wales":"wal","Cameroon":"cmr","Ghana":"gha","Morocco":"mar",
    "Senegal":"sen","Tunisia":"tun","Japan":"jpn","South Korea":"kor","Iran":"irn",
    "Qatar":"qat","Saudi Arabia":"ksa","Canada":"can","Mexico":"mex",
    "United States":"usa","Costa Rica":"crc","Australia":"aus",
}

# Lag (frames) para derivar velocidades por diferencias finitas (~0.5 s).
_VEL_LAG_FRAMES = 15
_SPEED_CAP_MPS  = 12.0          # cap anti-teleport del tracking


# ---- Adapter: frame de tracking PFF -> schema Z02 ----

def _load_frame_z02(match_id: int, frame_num: int,
                     vel_lag: int = _VEL_LAG_FRAMES) -> tuple:
    """Construye el DataFrame schema-Z02 de un frame de tracking PFF.

    Velocidades por diferencia finita contra el frame `vel_lag` atras.
    Devuelve (frame_df, ball_pos, att_team_id, meta).
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id, away_id = int(md["home_team_id"]), int(md["away_team_id"])
    home_name = str(md.get("home_team_name") or "")
    away_name = str(md.get("away_team_name") or "")
    date_iso = str(md.get("date") or "")[:10]   # YYYY-MM-DD
    comp = md.get("competition") or {}
    comp_name = str(comp.get("name") if isinstance(comp, dict) else "")
    fps = float(md.get("fps") or 29.97)
    pitch_l = float(md.get("pitch_length") or PITCH_LENGTH)
    pitch_w = float(md.get("pitch_width") or PITCH_WIDTH)
    dt = vel_lag / fps

    # Jerseys de portero por equipo (rosters)
    gk: set[tuple[int, int]] = set()
    for r in load_rosters(match_id).iter_rows(named=True):
        if r["position_group"] == "GK" and r["shirt_number"] is not None:
            gk.add((int(r["team_id"]), int(r["shirt_number"])))

    tr = scan_tracking(match_id).select([
        "frameNum",
        pl.col("homePlayersSmoothed").alias("home"),
        pl.col("awayPlayersSmoothed").alias("away"),
        pl.col("ballsSmoothed").alias("ball"),
    ]).filter(pl.col("frameNum").is_in([frame_num, frame_num - vel_lag])).collect()
    frames = {int(r["frameNum"]): r for r in tr.iter_rows(named=True)}
    cur = frames.get(frame_num)
    if cur is None:
        raise ValueError(f"frame {frame_num} ausente en match {match_id}")
    prev = frames.get(frame_num - vel_lag)

    def _xy(frame, side: str) -> dict[int, tuple[float, float]]:
        out = {}
        for p in (frame[side] or []):
            j, x = p.get("jerseyNum"), p.get("x")
            if j is not None and x is not None:
                out[int(j)] = (float(x), float(p["y"]))
        return out

    prev_xy = {"home": _xy(prev, "home") if prev else {},
                "away": _xy(prev, "away") if prev else {}}

    rows = []
    for side, tid in (("home", home_id), ("away", away_id)):
        for p in (cur[side] or []):
            j, x = p.get("jerseyNum"), p.get("x")
            if j is None or x is None:
                continue
            j, x, y = int(j), float(x), float(p["y"])
            px, py = prev_xy[side].get(j, (x, y))
            vx, vy = (x - px) / dt, (y - py) / dt
            sp = float(np.hypot(vx, vy))
            if sp > _SPEED_CAP_MPS:
                vx, vy = vx * _SPEED_CAP_MPS / sp, vy * _SPEED_CAP_MPS / sp
            rows.append(dict(x_tracking=x, y_tracking=y, vx=vx, vy=vy,
                              team_id=tid, is_ball=0,
                              is_goalkeeper=int((tid, j) in gk), jersey=j))
    ball = cur["ball"]
    if ball and ball.get("x") is not None:
        rows.append(dict(x_tracking=float(ball["x"]), y_tracking=float(ball["y"]),
                          vx=0.0, vy=0.0, team_id=-1, is_ball=1,
                          is_goalkeeper=0, jersey=-1))
    df = pd.DataFrame(rows)

    ball_pos = pc.get_ball_pos(df)
    field = df[df["is_ball"] == 0]
    d = np.hypot(field["x_tracking"] - ball_pos[0], field["y_tracking"] - ball_pos[1])
    att_team_id = int(field.iloc[int(d.values.argmin())]["team_id"])
    return df, ball_pos, att_team_id, dict(pitch_l=pitch_l, pitch_w=pitch_w,
                                             home_id=home_id, away_id=away_id,
                                             home_name=home_name, away_name=away_name,
                                             date_iso=date_iso, comp_name=comp_name)


def frame_for_clock(match_id: int, period: int, clock_s: float) -> int:
    """frameNum cuyo periodGameClockTime es el mas cercano a clock_s (su periodo)."""
    tr = scan_tracking(match_id).select(
        ["frameNum", "period", "periodGameClockTime"]
    ).filter(pl.col("period") == period).collect()
    idx = int((tr["periodGameClockTime"] - clock_s).abs().arg_min())
    return int(tr["frameNum"][idx])


# ---- Malla PPCF ----

def compute_ppcf_grid(frame_df: pd.DataFrame, att_team_id: int,
                       ball_pos: np.ndarray, pitch_l: float, pitch_w: float,
                       n_x: int = 80, n_y: int = 52) -> np.ndarray:
    """PPCF del equipo atacante sobre una malla n_y x n_x del campo (Z02)."""
    xs = np.linspace(-pitch_l / 2, pitch_l / 2, n_x)
    ys = np.linspace(-pitch_w / 2, pitch_w / 2, n_y)
    XX, YY = np.meshgrid(xs, ys)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    ppcf = pc.ppcf_at_targets(frame_df, targets, att_team_id, ball_pos)
    return ppcf.reshape(n_y, n_x)


# ---- Layout pitch-aligned (header + pitch + footer al mismo ancho) ----
# Portado de Diagonality_3D/viz/passes_plot.py. Campo aspect 105/68 = 1.544.
# Header / footer comparten ancho horizontal exacto con el pitch.

FIGSIZE = (14.0, 10.5)
PITCH_RATIO = PITCH_LENGTH / PITCH_WIDTH

H_HEADER = 0.10
H_PITCH = 0.68
H_FOOTER = 0.18
PAD_TOP = 0.015
PAD_HEADER_PITCH = 0.0
PAD_PITCH_FOOTER = 0.005
PAD_BOTTOM = 0.020


def _layout(figsize=FIGSIZE) -> dict:
    fw, fh = figsize
    pitch_h_in = H_PITCH * fh
    pitch_w_in = pitch_h_in * PITCH_RATIO
    pitch_w_frac = pitch_w_in / fw
    left = (1.0 - pitch_w_frac) / 2.0

    y_footer_bot = PAD_BOTTOM
    y_footer_top = y_footer_bot + H_FOOTER
    y_pitch_bot = y_footer_top + PAD_PITCH_FOOTER
    y_pitch_top = y_pitch_bot + H_PITCH
    y_header_bot = y_pitch_top + PAD_HEADER_PITCH
    return {
        "left": left, "width": pitch_w_frac,
        "header": (left, y_header_bot, pitch_w_frac, H_HEADER),
        "pitch":  (left, y_pitch_bot,  pitch_w_frac, H_PITCH),
        "footer": (left, y_footer_bot, pitch_w_frac, H_FOOTER),
    }


def _make_block_axes(fig, left, bottom, width, height):
    """Sub-axes limpio dentro del footer (sin spines, lim 0-1)."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# ---- Header ----

def _draw_header(ax: plt.Axes, title: str, subtitle,
                  team_logo_path: Optional[str] = None,
                  project_logo_path: Optional[str] = None) -> None:
    """Strip superior anclado al ancho del campo: escudo seleccion (izq),
    titulo + subtitulo, logo JO (derecha)."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    text_x = 0.135
    if team_logo_path and Path(team_logo_path).exists():
        try:
            img = plt.imread(str(team_logo_path))
            # zoom 0.70 escudo team — match passes_plot proporcion vs header
            ab = AnnotationBbox(OffsetImage(img, zoom=0.70),
                                 (0.0, 0.50), frameon=False,
                                 box_alignment=(0.0, 0.5))
            ab.set_clip_on(False)
            ax.add_artist(ab)
            text_x = 0.12
        except Exception:
            pass

    sub_lines = list(subtitle) if isinstance(subtitle, (list, tuple)) else [subtitle]
    n_sub = len(sub_lines)
    title_y = 0.82 if n_sub >= 2 else 0.55
    gap_title_sub = 0.36
    gap_sub_sub = 0.26

    ax.text(text_x, title_y, title, color=WHITE,
             fontsize=22, fontweight="bold", ha="left", va="center")
    y = title_y
    for i, line in enumerate(sub_lines):
        y -= gap_title_sub if i == 0 else gap_sub_sub
        # Subtitulos en BLANCO puro (sin alpha) — homogeneo con titulo
        ax.text(text_x, y, line, color=WHITE,
                 fontsize=13, ha="left", va="center")

    if project_logo_path and Path(project_logo_path).exists():
        try:
            img = plt.imread(str(project_logo_path))
            # zoom 0.12 logo JO (antes 0.08, muy pequeno)
            ab = AnnotationBbox(OffsetImage(img, zoom=0.12),
                                 (1.0, 0.42), frameon=False,
                                 box_alignment=(1.0, 0.5))
            ab.set_clip_on(False)
            ax.add_artist(ab)
        except Exception:
            pass


# ---- Footer (3 bloques: leyenda colormap + direccion + stats) ----

def _draw_footer(fig: plt.Figure, L: dict, att_team_name: str,
                  def_team_name: str, n_att: int, n_def: int,
                  match_clock: str, period: int,
                  attacking_right: bool = True) -> None:
    """Footer 3 bloques 40/20/40 del ancho del campo.

    Block 1 (40%): barra de gradiente PPCF + etiquetas de control.
    Block 2 (20%): triangulos direccion ataque + nombre del equipo atacante.
    Block 3 (40%): stats — N jugadores atacante / defensor + match clock.
    """
    fl = L["left"]; fw = L["width"]
    fb = L["footer"][1]; fh = L["footer"][3]
    b1_w, b2_w, b3_w = fw * 0.40, fw * 0.20, fw * 0.40
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ---- BLOCK 1: barra de gradiente PPCF horizontal + etiquetas ----
    bar_y = 0.65
    bar_h = 0.18
    bar_x0 = 0.10
    bar_x1 = 0.90
    n_steps = 200
    xs = np.linspace(bar_x0, bar_x1, n_steps + 1)
    for i in range(n_steps):
        c = PPCF_CMAP(i / (n_steps - 1))
        ax_b1.add_patch(mpatches.Rectangle(
            (xs[i], bar_y - bar_h / 2), xs[i + 1] - xs[i], bar_h,
            facecolor=c, edgecolor="none", transform=ax_b1.transAxes, zorder=2))
    # Borde fino + ticks
    ax_b1.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y - bar_h / 2), bar_x1 - bar_x0, bar_h,
        facecolor="none", edgecolor=WHITE, lw=0.7,
        transform=ax_b1.transAxes, zorder=3))
    # Etiquetas debajo del bar — TODAS en blanco
    ax_b1.text(bar_x0, bar_y - bar_h / 2 - 0.10, "100% defensor",
                ha="left", va="top", fontsize=10, color=WHITE,
                transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(0.5, bar_y - bar_h / 2 - 0.10, "neutro 50/50",
                ha="center", va="top", fontsize=10, color=WHITE,
                transform=ax_b1.transAxes)
    ax_b1.text(bar_x1, bar_y - bar_h / 2 - 0.10, "100% atacante",
                ha="right", va="top", fontsize=10, color=WHITE,
                transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(0.5, bar_y + bar_h / 2 + 0.18, "Pitch Control (PPCF)",
                ha="center", va="bottom", fontsize=11, color=WHITE,
                transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(0.5, bar_y + bar_h / 2 + 0.05,
                "probabilidad de que el equipo controle ese punto del campo",
                ha="center", va="bottom", fontsize=9, color=WHITE,
                transform=ax_b1.transAxes, style="italic")

    # ---- BLOCK 2: triangulos direccion ataque + nombre equipo ----
    n_tri = 4
    tri_w = 0.10
    tri_h = 0.16
    spacing = 0.06
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0
    y_tri = 0.62
    for i in range(n_tri):
        cx = start_x + i * (tri_w + spacing)
        if attacking_right:
            verts = [(cx, y_tri + tri_h / 2),
                      (cx + tri_w, y_tri),
                      (cx, y_tri - tri_h / 2)]
        else:
            verts = [(cx + tri_w, y_tri + tri_h / 2),
                      (cx, y_tri),
                      (cx + tri_w, y_tri - tri_h / 2)]
        alpha = 0.32 + i * 0.18
        ax_b2.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=ATT, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10))
    ax_b2.text(0.5, 0.36, f"{att_team_name} ataca",
                ha="center", va="center", fontsize=11, color=WHITE,
                transform=ax_b2.transAxes, fontweight="bold")
    ax_b2.text(0.5, 0.20, "direccion del juego",
                ha="center", va="center", fontsize=9, color=WHITE,
                transform=ax_b2.transAxes, style="italic")

    # ---- BLOCK 3: stats — N jugadores + reloj ----
    # 3 cells: ataque jugadores | defensa jugadores | reloj
    cx1, cx2, cx3 = 0.18, 0.52, 0.85
    y_num = 0.66
    y_lbl = 0.34
    ax_b3.text(cx1, y_num, f"{n_att}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=ATT,
                transform=ax_b3.transAxes)
    ax_b3.text(cx1, y_lbl, f"jugadores {att_team_name[:14]}",
                ha="center", va="center", fontsize=9, color=WHITE,
                transform=ax_b3.transAxes)
    ax_b3.text(cx2, y_num, f"{n_def}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=DEF,
                transform=ax_b3.transAxes)
    ax_b3.text(cx2, y_lbl, f"jugadores {def_team_name[:14]}",
                ha="center", va="center", fontsize=9, color=WHITE,
                transform=ax_b3.transAxes)
    ax_b3.text(cx3, y_num, match_clock, ha="center", va="center",
                fontsize=18, fontweight="bold", color=WHITE,
                transform=ax_b3.transAxes)
    ax_b3.text(cx3, y_lbl, f"periodo {period}  ·  reloj",
                ha="center", va="center", fontsize=9, color=WHITE,
                transform=ax_b3.transAxes)


# ---- Render principal ----

def plot_ppcf(match_id: int, frame_num: int, title: Optional[str] = None,
               subtitle=None, save_path=None,
               attacking_right: Optional[bool] = None) -> plt.Figure:
    """Render del frame: header + superficie PPCF + jugadores + footer.

    Titulo y subtitulo auto-generados desde metadata si no se especifican.
    """
    df, ball_pos, att, meta = _load_frame_z02(match_id, frame_num)
    grid = compute_ppcf_grid(df, att, ball_pos, meta["pitch_l"], meta["pitch_w"])

    # Nombres atacante / defensor
    att_team_name = meta["home_name"] if att == meta["home_id"] else meta["away_name"]
    def_team_name = meta["away_name"] if att == meta["home_id"] else meta["home_name"]
    att_slug = _TEAM_TO_SLUG.get(att_team_name)
    team_logo = (_LOGOS / f"{att_slug}.png") if att_slug else None

    # Attacking direction: mediana x del equipo atacante < ball.x => ataca a la dcha
    if attacking_right is None:
        att_x_med = float(df[(df["is_ball"] == 0) & (df["team_id"] == att)]
                            ["x_tracking"].median())
        attacking_right = att_x_med < ball_pos[0]

    # Match clock formateado + minuto absoluto (para titulo)
    tr_row = scan_tracking(match_id).select([
        "frameNum", "period", "periodGameClockTime"
    ]).filter(pl.col("frameNum") == frame_num).collect()
    period = int(tr_row["period"][0]) if tr_row.height else 0
    clock_s = float(tr_row["periodGameClockTime"][0]) if tr_row.height else 0.0
    mm, ss = int(clock_s // 60), int(clock_s % 60)
    clock_str = f"{mm:02d}:{ss:02d}"
    # Minuto absoluto del partido (periodo 2 anade ~45 min)
    minute_abs = mm + (45 if period >= 2 else 0)

    # Titulo auto desde metadata: "Pitch Control · {home} vs {away}"
    home_name = meta["home_name"]; away_name = meta["away_name"]
    if title is None:
        title = f"Pitch Control  ·  {home_name} vs {away_name}"
    if subtitle is None:
        comp = meta["comp_name"] or "Mundial Qatar 2022"
        subtitle = [
            f"{comp}  ·  min {minute_abs}  ·  reloj {clock_str} (P{period})",
            f"frame {frame_num}  ·  {meta['date_iso']}",
        ]

    # ---- Construye fig + 3 axes (header / pitch / footer) ----
    L = _layout(FIGSIZE)
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)

    ax_header = fig.add_axes(L["header"])
    _draw_header(ax_header, title, subtitle,
                  team_logo_path=(str(team_logo) if team_logo and team_logo.exists() else None),
                  project_logo_path=str(_LOGO_PATH) if _LOGO_PATH.exists() else None)

    ax_pitch = fig.add_axes(L["pitch"])
    draw_pitch(ax_pitch, meta["pitch_l"], meta["pitch_w"])
    L_h, W_h = meta["pitch_l"] / 2, meta["pitch_w"] / 2

    # Superficie PPCF
    ax_pitch.imshow(grid, extent=[-L_h, L_h, -W_h, W_h], origin="lower",
                     cmap=PPCF_CMAP, vmin=0, vmax=1, alpha=0.72,
                     interpolation="spline36", zorder=1, aspect="auto")

    field = df[df["is_ball"] == 0]
    # Flechas de velocidad (estilo Diagonality_3D ppcf_plot): mas visibles —
    # umbral bajo de 0.3 m/s (captura tambien jugadores en movimiento lento)
    # + scale=110 (mas grandes), alpha=0.85, width=0.005. Asi se entiende
    # quien empuja arriba y quien se cierra atras.
    for is_att, color in ((True, ATT), (False, DEF)):
        sub = field[(field["team_id"] == att) == is_att]
        sub = sub[np.hypot(sub["vx"], sub["vy"]) > 0.3]
        if len(sub):
            ax_pitch.quiver(sub["x_tracking"], sub["y_tracking"],
                              sub["vx"], sub["vy"], color=color, scale=110,
                              scale_units="width", width=0.005, headwidth=4.0,
                              headlength=4.5, headaxislength=4.0, alpha=0.85,
                              zorder=3)
    # Jugadores + dorsales
    for _, p in field.iterrows():
        color = (GK if p["is_goalkeeper"]
                  else (ATT if p["team_id"] == att else DEF))
        ax_pitch.plot(p["x_tracking"], p["y_tracking"], "o", ms=22, color=color,
                       markeredgecolor=WHITE, markeredgewidth=1.3, alpha=0.93,
                       zorder=5)
        ax_pitch.text(p["x_tracking"], p["y_tracking"], str(int(p["jersey"])),
                       color=WHITE, fontsize=8.5, ha="center", va="center",
                       fontweight="bold", zorder=6, path_effects=PE_S)
    # Balon
    ax_pitch.plot(ball_pos[0], ball_pos[1], "o", ms=11, color=BALL,
                    markeredgecolor="black", markeredgewidth=0.9, zorder=10)

    n_att = int((field["team_id"] == att).sum())
    n_def = int((field["team_id"] != att).sum())
    _draw_footer(fig, L,
                  att_team_name=att_team_name, def_team_name=def_team_name,
                  n_att=n_att, n_def=n_def,
                  match_clock=clock_str, period=period,
                  attacking_right=attacking_right)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, facecolor=BG, bbox_inches=None)
    return fig


# ---- Sanity / hero figure ----

if __name__ == "__main__":
    # CLI generico: <match_id> <period> <minute_in_period>
    # ej:  python -m src.viz.ppcf 3835 2 18    (ARG-MEX, P2 min 18 = 63' total)
    # ej:  python -m src.viz.ppcf              (default Messi ARG-MEX)
    if len(sys.argv) >= 4:
        mid = int(sys.argv[1]); per = int(sys.argv[2])
        minute = float(sys.argv[3])
        fnum = frame_for_clock(mid, period=per, clock_s=minute * 60.0)
        slug = f"{mid}_p{per}_m{int(minute)}"
    else:
        mid = 3835; per = 2; minute = 63.45
        fnum = frame_for_clock(mid, period=per, clock_s=minute * 60.0)
        slug = "messi_arg_mex"
    out = f"outputs/viz/ppcf_{slug}.png"
    print(f"[ppcf] match {mid}  P{per} min {minute}  frame = {fnum}")
    plot_ppcf(mid, fnum, save_path=out)
    print(f"OK -> {out}")
