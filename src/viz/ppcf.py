"""ppcf - Superficie Pitch Control (Spearman 2018) sobre el campo.

Layout pitch-aligned: la figura tiene EXACTAMENTE el ancho del campo. Header
(escudo del equipo atacante + titulo auto desde metadata + JO) y footer
(barra gradiente PPCF + flechas direccion + leyenda equipos + balon Telstar)
ocupan el mismo ancho horizontal que el pitch.

Z02_pitch_control hace el computo del PPCF. Este modulo:
  - Adapter del frame de tracking PFF (25/30 Hz) al schema Z02
  - Auto-correccion del espejo eventos<->tracking en prorroga (PFF re-origina P3/P4)
  - Render con identidad LIGHT OPTA (fondo blanco, ATT/DEF light, balon Telstar)

Uso:
    python -m src.viz.ppcf <match_id> <match_minute>
    python -m src.viz.ppcf 10517 81              # final ARG-FRA, 2-2 de Mbappe
    python -m src.viz.ppcf                       # default: 2-2 de Mbappe (final)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches                              # Rectangle/Polygon/Ellipse/Circle del footer y balon

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import Z02_pitch_control as pc                                      # computo PPCF (Spearman 2018 vectorizado)
from M01_loader_pff import scan_tracking, load_metadata, load_rosters
from M03_preprocess import attacking_direction, goals_timeline

from viz.common import (ATT, BG, DEF, GK, LEGEND, PE_S, PITCH_LENGTH, PITCH_WIDTH,
                         PPCF_CMAP, TOURNAMENT_ES, WHITE, draw_pitch, draw_header,
                         team_es)

_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"               # escudos selecciones (sportlogos/sport.db.logos)

# pff team_name -> iso3 slug (sportlogos/sport.db.logos). Necesario pa
# escoger el escudo del equipo atacante en el header.
_TEAM_TO_SLUG = {
    "Argentina":"arg","Brazil":"bra","Ecuador":"ecu","Uruguay":"uru","Belgium":"bel",
    "Croatia":"cro","Denmark":"den","England":"eng","France":"fra","Germany":"ger",
    "Netherlands":"ned","Poland":"pol","Portugal":"por","Serbia":"srb","Spain":"esp",
    "Switzerland":"sui","Wales":"wal","Cameroon":"cmr","Ghana":"gha","Morocco":"mar",
    "Senegal":"sen","Tunisia":"tun","Japan":"jpn","South Korea":"kor","Iran":"irn",
    "Qatar":"qat","Saudi Arabia":"ksa","Canada":"can","Mexico":"mex",
    "United States":"usa","Costa Rica":"crc","Australia":"aus",
}

_VEL_LAG_FRAMES = 15                                                # ~0.5 s a 30 fps — lag pa diff finita de velocidades
_SPEED_CAP_MPS  = 12.0                                              # cap anti-teleport del tracking PFF


# ----------------------------------------------------------------------------
# Adapter: frame de tracking PFF -> schema Z02
# ----------------------------------------------------------------------------

def _load_frame_z02(match_id: int, frame_num: int,
                     vel_lag: int = _VEL_LAG_FRAMES) -> tuple:
    """Construye el DataFrame Z02 de un frame de tracking PFF.

    Velocidades por diferencia finita contra el frame `vel_lag` atras.
    Devuelve (frame_df, ball_pos, att_team_id, meta).
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id, away_id = int(md["home_team_id"]), int(md["away_team_id"])
    home_name = str(md.get("home_team_name") or "")
    away_name = str(md.get("away_team_name") or "")
    date_iso = str(md.get("date") or "")[:10]                       # YYYY-MM-DD
    comp = md.get("competition") or {}
    comp_name = str(comp.get("name") if isinstance(comp, dict) else "")
    fps = float(md.get("fps") or 29.97)
    pitch_l = float(md.get("pitch_length") or PITCH_LENGTH)
    pitch_w = float(md.get("pitch_width") or PITCH_WIDTH)
    dt = vel_lag / fps                                              # delta_t entre frames cur y prev (segundos)

    # Jerseys de portero por equipo (pa colorear distinto en el render)
    gk: set[tuple[int, int]] = set()
    for r in load_rosters(match_id).iter_rows(named=True):
        if r["position_group"] == "GK" and r["shirt_number"] is not None:
            gk.add((int(r["team_id"]), int(r["shirt_number"])))

    # Pillo frame actual + el de hace vel_lag (pa velocidades)
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
        """jersey -> (x, y) de un equipo en un frame."""
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
            px, py = prev_xy[side].get(j, (x, y))                   # si no estaba antes, vel=0
            vx, vy = (x - px) / dt, (y - py) / dt
            sp = float(np.hypot(vx, vy))
            if sp > _SPEED_CAP_MPS:                                 # capa teleports del tracking
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

    # ---- Equipo atacante = POSESION REAL (game_event.home_ball) ----
    # En frames de tracking, home_ball viene null entre eventos, asi que
    # tomamos el valor no-null mas cercano (la posesion vigente del balon).
    poss = scan_tracking(match_id).select([
        "frameNum", "period",
        pl.col("game_event").struct.field("home_ball").alias("hb"),
    ]).filter(pl.col("frameNum").is_between(frame_num - 250, frame_num + 250)).collect()
    period = int(poss.filter(pl.col("frameNum") == frame_num)["period"][0])
    nn = (poss.filter(pl.col("hb").is_not_null())
               .with_columns((pl.col("frameNum") - frame_num).abs().alias("d"))
               .sort("d"))
    if nn.height == 0:
        raise ValueError(f"sin posesion (home_ball) cerca del frame {frame_num}")
    att_team_id = home_id if bool(nn["hb"][0]) else away_id

    # ---- Correccion del espejo eventos<->tracking en prorroga ----
    # PFF entrega, en algunos partidos de prorroga (P3/P4), las coordenadas de
    # tracking en un frame ROTADO 180 respecto al de los eventos. attacking_direction
    # vive en el frame de eventos (verdad fisica: casa con disparos y goles). Si la
    # posicion del portero en el tracking implica la direccion CONTRARIA a la de
    # eventos, el frame de tracking esta rotado -> lo des-roto (x,y -> -x,-y) pa
    # que todo viva en el frame de eventos. Auto-detectado por partido/periodo.
    ev_dir = attacking_direction(match_id).filter(
        (pl.col("team_id") == att_team_id) & (pl.col("period") == period))
    ev_right = (str(ev_dir["direction"][0]) == "R") if ev_dir.height else True
    tr_right = _attacking_right_tracking(match_id, att_team_id, period, home_id)
    if ev_right != tr_right:
        # PFF re-origina el frame de tracking en prorroga: rotacion 180 (x,y -> -x,-y)
        # Verificado por jugador (corr_X y corr_Y < 0 en ET).
        df[["x_tracking", "y_tracking", "vx", "vy"]] *= -1.0

    ball_pos = pc.get_ball_pos(df)
    return df, ball_pos, att_team_id, dict(pitch_l=pitch_l, pitch_w=pitch_w,
                                             home_id=home_id, away_id=away_id,
                                             home_name=home_name, away_name=away_name,
                                             date_iso=date_iso, comp_name=comp_name,
                                             period=period)


def _attacking_right_tracking(match_id: int, att_team_id: int, period: int,
                               home_id: int) -> bool:
    """Direccion de ataque del equipo en posesion, derivada del TRACKING.

    El portero del equipo atacante defiende su porteria: si su x mediana en el
    periodo < 0 (porteria izquierda) el equipo ataca hacia la derecha (+x).
    Robusto al espejo eventos/tracking que PFF mete en prorroga: mide sobre el
    mismo frame que se pinta.
    """
    side = "home" if att_team_id == home_id else "away"
    gk_jn = {str(int(r["shirt_number"]))
             for r in load_rosters(match_id).iter_rows(named=True)
             if r["position_group"] == "GK" and int(r["team_id"]) == att_team_id
             and r["shirt_number"] is not None}
    tr = scan_tracking(match_id).select([
        "period", pl.col(f"{side}PlayersSmoothed").alias("p"),
    ]).filter(pl.col("period") == period).collect()
    by_jersey: dict[str, list[float]] = {}
    for r in tr.iter_rows(named=True):
        for pp in (r["p"] or []):
            if pp and str(pp.get("jerseyNum")) in gk_jn and pp.get("x") is not None:
                by_jersey.setdefault(str(pp["jerseyNum"]), []).append(float(pp["x"]))
    if not by_jersey:
        return True
    # Portero titular = jersey con |mediana x| mayor (el pegado a una porteria)
    starter = max(by_jersey.values(), key=lambda v: abs(np.median(v)))
    return float(np.median(starter)) < 0


def frame_for_clock(match_id: int, period: int, clock_s: float) -> int:
    """frameNum cuyo periodGameClockTime es el mas cercano a clock_s (en su periodo)."""
    tr = scan_tracking(match_id).select(
        ["frameNum", "period", "periodGameClockTime"]
    ).filter(pl.col("period") == period).collect()
    idx = int((tr["periodGameClockTime"] - clock_s).abs().arg_min())
    return int(tr["frameNum"][idx])


# ----------------------------------------------------------------------------
# Malla PPCF
# ----------------------------------------------------------------------------

def compute_ppcf_grid(frame_df: pd.DataFrame, att_team_id: int,
                       ball_pos: np.ndarray, pitch_l: float, pitch_w: float,
                       n_x: int = 80, n_y: int = 52) -> np.ndarray:
    """PPCF del equipo atacante en una malla n_y x n_x del campo (via Z02).

    ↑ n_x, n_y → malla mas densa = grafico mas suave pero mas lento.
    """
    xs = np.linspace(-pitch_l / 2, pitch_l / 2, n_x)
    ys = np.linspace(-pitch_w / 2, pitch_w / 2, n_y)
    XX, YY = np.meshgrid(xs, ys)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    ppcf = pc.ppcf_at_targets(frame_df, targets, att_team_id, ball_pos)
    return ppcf.reshape(n_y, n_x)


# ----------------------------------------------------------------------------
# Layout pitch-aligned
# ----------------------------------------------------------------------------
# La figura tiene EXACTAMENTE el ancho del campo, sin margenes laterales.
# El alto se deriva pa que el axes del pitch tenga el ratio xlim/ylim correcto:
#   xlim = [-55.5, 55.5] -> rango 111 m (PITCH_LENGTH + 6 de margen)
#   ylim = [-37,   37  ] -> rango 74 m  (PITCH_WIDTH  + 6 de margen)
#   ratio datos = 111/74 = 1.5
# Con axes width=FIG_W y height=FIG_W/1.5, set_aspect("equal") encaja sin huecos.

FIG_W            = 14.0    # ancho fijo en pulgadas (= ancho fisico del campo en la viz)
HEADER_H_IN      = 1.1     # alto del header en pulgadas — ↑ header MAS ALTO / ↓ mas espacio pal campo
FOOTER_H_IN      = 1.3     # alto del footer (leyenda) — ↑ MAS ALTO
PAD_TOP_IN       = 0.04    # margen entre tope de fig y header (pulgadas)
PAD_HDR_PITCH_IN = 0.0     # separacion header-campo (0 = pegados)
PAD_PCH_FTR_IN   = 0.05    # separacion campo-footer
PAD_BOT_IN       = 0.07    # margen entre footer y base
PITCH_MARGIN_Y_M = 1.0     # margen Y del axes del campo (arriba/abajo) — antes 3
                            # ↓ campo MAS PEGADO al header/footer (menos hueco)
                            # ↑ MAS espacio entre lineas del campo y bordes del axes
PITCH_MARGIN_X_M = 2.0     # margen X del axes (laterales) — MIN 1.5 pa que entren porterias (±54m)
                            # ↑ MAS espacio lateral (mas aire para porterias y nombres de equipo)

_DATA_RATIO = (PITCH_LENGTH + 2 * PITCH_MARGIN_X_M) / (PITCH_WIDTH + 2 * PITCH_MARGIN_Y_M)
PITCH_H_IN  = FIG_W / _DATA_RATIO                                    # alto del axes del campo (pulgadas)

FIG_H = (PAD_TOP_IN + HEADER_H_IN + PAD_HDR_PITCH_IN
         + PITCH_H_IN + PAD_PCH_FTR_IN
         + FOOTER_H_IN + PAD_BOT_IN)

FIGSIZE = (FIG_W, FIG_H)


def _layout(figsize: tuple = FIGSIZE) -> dict:
    """Posiciones absolutas de header/pitch/footer. Pitch ocupa todo el ancho."""
    _, fh = figsize
    h_hdr = HEADER_H_IN      / fh                                    # alto header en fraccion de figura
    h_pch = PITCH_H_IN       / fh                                    # alto pitch
    h_ftr = FOOTER_H_IN      / fh                                    # alto footer
    p_hp  = PAD_HDR_PITCH_IN / fh                                    # pad header-pitch
    p_pf  = PAD_PCH_FTR_IN   / fh                                    # pad pitch-footer
    p_bot = PAD_BOT_IN       / fh                                    # pad bottom

    y_ftr_bot = p_bot                                                # footer arranca encima del bottom margin
    y_pch_bot = y_ftr_bot + h_ftr + p_pf                             # pitch arranca encima del footer
    y_pch_top = y_pch_bot + h_pch
    y_hdr_bot = y_pch_top + p_hp                                     # header arranca encima del pitch
    return {
        "left": 0.0, "width": 1.0,                                   # ocupan todo el ancho
        "header": (0.0, y_hdr_bot, 1.0, h_hdr),                      # (x, y, w, h) en fraccion de figura
        "pitch":  (0.0, y_pch_bot, 1.0, h_pch),
        "footer": (0.0, y_ftr_bot, 1.0, h_ftr),
    }


def _make_block_axes(fig, left, bottom, width, height):
    """Sub-axes limpio (sin spines, sin ticks, lim 0-1). Pa los 3 bloques del footer."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)                             # data coords == axes fraction
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# ----------------------------------------------------------------------------
# Balon Telstar (pentagonos sobre circulo blanco)
# ----------------------------------------------------------------------------

def _draw_football_on_pitch(ax: plt.Axes, cx: float, cy: float,
                             r: float = 0.65, zorder: int = 10) -> None:
    """Balon de futbol patron Telstar en coords del campo (metros).

    El campo debe tener set_aspect('equal') pa que el circulo no se deforme.
      r:        radio del balon en metros — ↑ MAS GRANDE
      zorder:   profundidad del balon (>jugadores=5, >dorsales=6)
    """
    n = 5                                                            # pentagonos = 5 lados
    # Circulo blanco con borde negro = balon base
    ball = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor="black",
                            linewidth=1.5, zorder=zorder)
    ax.add_patch(ball)
    # Pentagono central negro (r * 0.37 → ocupa 37% del radio)
    rp = r * 0.37
    c_ang = np.array([np.pi / 2 + 2 * np.pi * i / n for i in range(n)])
    pts_c = np.column_stack([cx + rp * np.cos(c_ang), cy + rp * np.sin(c_ang)])
    ax.add_patch(mpatches.Polygon(pts_c, facecolor="black", edgecolor="none",
                                   zorder=zorder + 1))
    # 5 pentagones exteriores: vertice apuntando AL borde (touching circumference)
    r_outer, rp2 = r * 0.73, r * 0.27                                # 0.73+0.27 = 1.0 → vertice toca el borde
    for i in range(n):
        ang = c_ang[i]
        ox = cx + r_outer * np.cos(ang)
        oy = cy + r_outer * np.sin(ang)
        # Pentagono exterior orientado pa que un vertice apunte hacia fuera
        o_ang = np.array([ang + 2 * np.pi * k / n for k in range(n)])
        pts_o = np.column_stack([ox + rp2 * np.cos(o_ang), oy + rp2 * np.sin(o_ang)])
        patch = mpatches.Polygon(pts_o, facecolor="black", edgecolor="none",
                                  zorder=zorder + 1)
        patch.set_clip_path(ball)                                    # recorta lo que se salga del circulo del balon
        ax.add_patch(patch)


# ----------------------------------------------------------------------------
# Footer — 3 bloques 40/20/40 del ancho del pitch
# ----------------------------------------------------------------------------

def _draw_footer(fig: plt.Figure, L: dict, att_team_name: str,
                  def_team_name: str, attacking_right: bool = True) -> None:
    """Footer 3 bloques:
      block 1 (40%) — barra gradiente PPCF + etiquetas de control
      block 2 (20%) — triangulos Direccion de Ataque
      block 3 (40%) — leyenda equipos (atacante azul + defensor rojo) + balon Telstar
    """
    fl = L["left"]; fw = L["width"]
    fb = L["footer"][1]; fh = L["footer"][3]
    b1_w, b2_w, b3_w = fw * 0.40, fw * 0.20, fw * 0.40              # split 40/20/40
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ---- BLOCK 1: barra gradiente horizontal del PPCF + etiquetas ----
    bar_y  = 0.75                                                    # y de la barra en fraccion del bloque — ↑ SUBE
    bar_h  = 0.20                                                    # alto de la barra — ↑ MAS ALTA
    bar_x0 = 0.10                                                    # extremo izq — ↑ MAS a la DERECHA
    bar_x1 = 0.90                                                    # extremo dcha — ↑ MAS a la DERECHA
    n_steps = 200                                                    # ↑ MAS pasos → gradiente MAS SUAVE
    xs = np.linspace(bar_x0, bar_x1, n_steps + 1)
    for i in range(n_steps):
        c = PPCF_CMAP(i / (n_steps - 1))                             # color en cada paso del gradiente
        ax_b1.add_patch(mpatches.Rectangle(
            (xs[i], bar_y - bar_h / 2), xs[i + 1] - xs[i], bar_h,
            facecolor=c, edgecolor="none", transform=ax_b1.transAxes, zorder=2))
    # Borde fino alrededor de la barra (encima del gradiente)
    ax_b1.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y - bar_h / 2), bar_x1 - bar_x0, bar_h,
        facecolor="none", edgecolor=WHITE, lw=0.7,
        transform=ax_b1.transAxes, zorder=3))
    # Etiquetas extremos debajo de la barra
    ax_b1.text(bar_x0, bar_y - bar_h / 2 - 0.1, "100% defensor",
                ha="left", va="top", fontsize=14, color=LEGEND,
                transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(bar_x1, bar_y - bar_h / 2 - 0.1, "100% atacante",
                ha="right", va="top", fontsize=14, color=LEGEND,
                transform=ax_b1.transAxes, fontweight="bold")
    # Descripcion arriba (normal: ni bold ni cursiva)
    ax_b1.text(0.5, bar_y + bar_h / 2 + 0.05,
                "probabilidad de que el equipo controle ese punto del campo",
                ha="center", va="bottom", fontsize=12, color=LEGEND,
                transform=ax_b1.transAxes)

    # ---- BLOCK 2: triangulos Direccion de Ataque ----
    n_tri   = 4                                                      # ↑ MAS triangulos / ↓ menos
    tri_w   = 0.10                                                   # ↑ triangulo MAS ANCHO
    tri_h   = 0.16                                                   # ↑ triangulo MAS ALTO
    spacing = 0.06                                                   # ↑ MAS hueco entre triangulos
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0                                  # centra los triangulos en el bloque
    y_tri   = 0.76                                                   # y de los triangulos — ↑ SUBEN
    for i in range(n_tri):
        cx = start_x + i * (tri_w + spacing)
        if attacking_right:                                          # triangulos apuntando a la DERECHA
            verts = [(cx, y_tri + tri_h / 2),
                      (cx + tri_w, y_tri),
                      (cx, y_tri - tri_h / 2)]
        else:                                                        # apuntando a la IZQUIERDA
            verts = [(cx + tri_w, y_tri + tri_h / 2),
                      (cx, y_tri),
                      (cx + tri_w, y_tri - tri_h / 2)]
        alpha = 0.32 + i * 0.18                                      # ↑ base/step → triangulos MAS OPACOS (degrade)
        ax_b2.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=ATT, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10))
    # Etiqueta unica debajo de los triangulos
    ax_b2.text(0.5, 0.5, "Dirección de Ataque",
                ha="center", va="center", fontsize=14, color=LEGEND,
                transform=ax_b2.transAxes, fontweight="bold")

    # ---- BLOCK 3: equipos (izq + centro) + balon Telstar (dcha) ----
    y_node = 0.75                                                    # y comun pa los 3 elementos
    # Nodos de los 2 equipos: marker "o" con ms=25 (= 25pt diametro)
    for x_node, x_txt, color, name in (
            (0.07, 0.125, ATT, att_team_name),                       # IZQUIERDA: atacante (azul)
            (0.35, 0.405, DEF, def_team_name)):                      # CENTRO:    defensor (rojo)
        ax_b3.plot(x_node, y_node, "o", ms=25, color=color, markeredgecolor=WHITE,
                    markeredgewidth=1.3, alpha=0.93, transform=ax_b3.transAxes,
                    clip_on=False, zorder=5)
        ax_b3.text(x_txt, y_node, team_es(name), ha="left", va="center", fontsize=14,
                    color=LEGEND, fontweight="bold", transform=ax_b3.transAxes)
    # ---- Balon Telstar (dcha) ----
    # Mismo patron que en el campo, pero usando Ellipse (no Circle) pa compensar
    # que ax_b3 es no-cuadrado (5.6" wide x 1.3" tall → aspect ~4.31).
    x_ball = 0.70                                                    # x del balon en fraccion del bloque
    _b3_aspect = (FIG_W * b3_w) / FOOTER_H_IN                        # ≈ 5.6 / 1.3 ≈ 4.31
    # ms=25pt en nodos → diametro 25pt = 0.347" = 0.267 frac alto → radio y = 0.134
    _r_y = 0.134                                                     # radio Y en fraccion del bloque → mismo tamano visible que los nodos
    _r_x = _r_y / _b3_aspect                                         # radio X compensado pa que la elipse parezca circulo
    _n = 5
    _c_ang = np.array([np.pi / 2 + 2 * np.pi * i / _n for i in range(_n)])
    _ball_ell = mpatches.Ellipse((x_ball, y_node), 2 * _r_x, 2 * _r_y,
                                   facecolor="white", edgecolor="black",
                                   linewidth=1.5, zorder=5)
    ax_b3.add_patch(_ball_ell)
    # Pentagono central negro
    _pts_c = np.column_stack([x_ball + _r_x * 0.37 * np.cos(_c_ang),
                                y_node + _r_y * 0.37 * np.sin(_c_ang)])
    _cp = mpatches.Polygon(_pts_c, facecolor="black", edgecolor="none", zorder=6)
    _cp.set_clip_path(_ball_ell); ax_b3.add_patch(_cp)
    # 5 pentagones exteriores (vertice tocando el borde)
    for _i in range(_n):
        _ang = _c_ang[_i]
        _ox = x_ball + _r_x * 0.73 * np.cos(_ang)
        _oy = y_node + _r_y * 0.73 * np.sin(_ang)
        _o_ang = np.array([_ang + 2 * np.pi * _k / _n for _k in range(_n)])
        _pts_o = np.column_stack([_ox + _r_x * 0.27 * np.cos(_o_ang),
                                    _oy + _r_y * 0.27 * np.sin(_o_ang)])
        _op = mpatches.Polygon(_pts_o, facecolor="black", edgecolor="none", zorder=6)
        _op.set_clip_path(_ball_ell); ax_b3.add_patch(_op)
    # Label "Balón" a la derecha del balon (mismo gap que los nodos: +0.075)
    ax_b3.text(x_ball + 0.055, y_node, "Balón", ha="left", va="center",
                fontsize=14, color=LEGEND, fontweight="bold",
                transform=ax_b3.transAxes)


# ----------------------------------------------------------------------------
# Render principal
# ----------------------------------------------------------------------------

def plot_ppcf(match_id: int, frame_num: int, title: Optional[str] = None,
               subtitle=None, save_path=None,
               attacking_right: Optional[bool] = None) -> plt.Figure:
    """Render del frame: header + superficie PPCF + jugadores + footer.

    Si title/subtitle son None, se generan automaticamente desde metadata
    (formato Opta: "Pitch Control: Home vs Away" + "Competicion | H X - Y A (d de mes de YYYY)").
    """
    df, ball_pos, att, meta = _load_frame_z02(match_id, frame_num)
    grid = compute_ppcf_grid(df, att, ball_pos, meta["pitch_l"], meta["pitch_w"])

    # Nombres atacante / defensor + escudo del equipo atacante (pal header)
    att_team_name = meta["home_name"] if att == meta["home_id"] else meta["away_name"]
    def_team_name = meta["away_name"] if att == meta["home_id"] else meta["home_name"]
    att_slug = _TEAM_TO_SLUG.get(att_team_name)
    team_logo = (_LOGOS / f"{att_slug}.png") if att_slug else None

    # Direccion de ataque desde attacking_direction (frame de eventos = verdad
    # fisica). El tracking ya se des-espeja en _load_frame_z02 pa vivir en este
    # mismo frame, asi que flecha y posiciones casan. 'R' -> ataca a la derecha.
    if attacking_right is None:
        d = attacking_direction(match_id).filter(
            (pl.col("team_id") == att) & (pl.col("period") == meta["period"]))
        attacking_right = (str(d["direction"][0]) == "R") if d.height else True

    # Marcador final del partido (pa el subtitulo del header)
    g = goals_timeline(match_id)
    score_home = int(g["cum_home"].max() or 0) if g.height else 0
    score_away = int(g["cum_away"].max() or 0) if g.height else 0

    # Titulo + subtitulo auto-generados desde metadata (formato Opta-style)
    # Naming ESTANDAR (todo español): nombres equipos traducidos via team_es(),
    # torneo siempre TOURNAMENT_ES de common.py, marcador X-Y sin espacios, fecha en parens.
    home_es = team_es(meta["home_name"])
    away_es = team_es(meta["away_name"])
    if title is None:
        title = f"Pitch Control: {home_es} vs {away_es}"
    if subtitle is None:
        from datetime import datetime
        _MES = ["enero","febrero","marzo","abril","mayo","junio",
                "julio","agosto","septiembre","octubre","noviembre","diciembre"]
        try:
            dt = datetime.strptime(meta["date_iso"], "%Y-%m-%d")
            date_fmt = f"{dt.day} de {_MES[dt.month-1]} de {dt.year}"
        except Exception:
            date_fmt = meta["date_iso"]
        subtitle = (f"{TOURNAMENT_ES}  |  "
                    f"{home_es} {score_home}-{score_away} {away_es}"
                    f"  ({date_fmt})")

    # ---- Construye fig + axes (header se pinta como artists, no axes) ----
    L = _layout(FIGSIZE)
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)

    # Header reusable de common (escudo IZQ + titulo+sub + JO DCHA), anclado a la franja del layout
    draw_header(fig, title=title,
                subtitle=subtitle if isinstance(subtitle, str) else None,
                escudo_path=(str(team_logo) if team_logo and team_logo.exists() else None),
                hdr_band=(L["header"][1], L["header"][1] + L["header"][3]),
                sub_size=14)

    # ---- Pitch ----
    ax_pitch = fig.add_axes(L["pitch"])
    draw_pitch(ax_pitch, meta["pitch_l"], meta["pitch_w"],
                margin=PITCH_MARGIN_Y_M, margin_x=PITCH_MARGIN_X_M)
    L_h, W_h = meta["pitch_l"] / 2, meta["pitch_w"] / 2

    # ---- Superficie PPCF sobre el campo ----
    # alpha=0.72 → PPCF semi-transparente pa que se vean las lineas del campo debajo
    # interpolation="spline36" → suavizado bilinear+ (visible pero no artificial)
    # vmin=0/vmax=1 → mapea PPCF directo al cmap (0=defensor 100%, 1=atacante 100%)
    ax_pitch.imshow(grid, extent=[-L_h, L_h, -W_h, W_h], origin="lower",
                     cmap=PPCF_CMAP, vmin=0, vmax=1, alpha=0.72,
                     interpolation="spline36", zorder=1, aspect="auto")

    field = df[df["is_ball"] == 0]

    # ---- Flechas de velocidad por equipo ----
    # ↓ umbral 0.3 m/s captura tambien jugadores lentos (vs 0.6 que ocultaba muchos)
    # ↑ scale -> flechas MAS PEQUENAS; ↓ scale -> MAS GRANDES
    # ↑ width -> trazo MAS GRUESO; ↑ alpha -> MAS OPACO sobre el PPCF
    for is_att, color in ((True, ATT), (False, DEF)):
        sub = field[(field["team_id"] == att) == is_att]
        sub = sub[np.hypot(sub["vx"], sub["vy"]) > 0.3]
        if len(sub):
            ax_pitch.quiver(sub["x_tracking"], sub["y_tracking"],
                              sub["vx"], sub["vy"], color=color, scale=110,
                              scale_units="width", width=0.005, headwidth=4.0,
                              headlength=4.5, headaxislength=4.0, alpha=0.85,
                              zorder=3)

    # ---- Jugadores (circulo grande + dorsal BLANCO con halo NEGRO) ----
    # ms=25 → mismo diametro que los nodos de la leyenda (25pt); ↑ → jugador MAS GRANDE
    # Dorsal: color="white" + PE_S (halo negro) → maxima legibilidad sobre azul/rojo saturados
    # markeredgewidth=1.3 → borde negro fino; ↑ → borde MAS GRUESO
    for _, p in field.iterrows():
        color = (GK if p["is_goalkeeper"]
                  else (ATT if p["team_id"] == att else DEF))
        ax_pitch.plot(p["x_tracking"], p["y_tracking"], "o", ms=25, color=color,
                       markeredgecolor=WHITE, markeredgewidth=1.3, alpha=0.93,
                       zorder=5)
        ax_pitch.text(p["x_tracking"], p["y_tracking"], str(int(p["jersey"])),
                       color="white", fontsize=12, ha="center", va="center",
                       fontweight="bold", zorder=6, path_effects=PE_S)

    # ---- Balon Telstar ----
    _draw_football_on_pitch(ax_pitch, ball_pos[0], ball_pos[1])

    # ---- Footer (gradiente PPCF + direccion + leyenda + balon) ----
    _draw_footer(fig, L,
                  att_team_name=att_team_name, def_team_name=def_team_name,
                  attacking_right=attacking_right)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches=None (NO 'tight') porque el layout esta absoluto en fig coords
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
    return fig


# ----------------------------------------------------------------------------
# CLI: python -m src.viz.ppcf [match_id] [match_minute]
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # python -m src.viz.ppcf <match_id> <match_minute>
    # ej:  python -m src.viz.ppcf 10517 81   (final, minuto 81 = 2-2 de Mbappe)
    # ej:  python -m src.viz.ppcf            (default 2-2 de Mbappe, final, min 81)
    if len(sys.argv) >= 3:
        mid = int(sys.argv[1]); match_min = float(sys.argv[2])
        slug = f"{mid}_min{int(match_min)}"
    else:
        mid, match_min = 10517, 80.97         # final ARG-FRA, 2-2 de Mbappe (volea, P2)
        slug = "mbappe_2_2_final"
    # Periodo: si el minuto cae en P1, P1; si no, P2 (no manejamos prorroga desde CLI)
    md = load_metadata(mid).row(0, named=True)
    end_p1 = float(md.get("endPeriod1") or 3000.0) / 60.0
    period = 1 if match_min <= end_p1 + 1 else 2
    fnum = frame_for_clock(mid, period=period, clock_s=match_min * 60.0)
    out = f"outputs/viz/ppcf_{slug}.png"
    print(f"[ppcf] match {mid}  min {match_min} (P{period})  frame = {fnum}")
    plot_ppcf(mid, fnum, save_path=out)
    print(f"OK -> {out}")
