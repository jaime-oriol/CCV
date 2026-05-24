"""ppcf - Superficie Pitch Control (Spearman 2018) sobre el campo.

Layout pitch-aligned: header + pitch + footer comparten el MISMO ancho
horizontal (el del campo). Header con escudo de la seleccion atacante,
titulo auto-generado desde metadata, logo JO. Footer 3 bloques: gradiente
PPCF + direccion del juego + leyenda de nodos por equipo.

Z02_pitch_control hace el computo del PPCF; este modulo arma la malla del
campo, el adapter del frame de tracking PFF 25/30 Hz al schema Z02, y el
render con identidad visual propia.

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
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import Z02_pitch_control as pc
from M01_loader_pff import scan_tracking, load_metadata, load_rosters
from M03_preprocess import attacking_direction, goals_timeline

from viz.common import (ATT, BALL, BG, DEF, GK, PE_S, PITCH_LENGTH,
                         PITCH_WIDTH, PPCF_CMAP, WHITE, draw_pitch, draw_header,
                         _LOGO_PATH)

_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"

# pff team_name -> iso3 slug (sportlogos/sport.db.logos). Necesario para
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

_VEL_LAG_FRAMES = 15            # ~0.5 s a 30 fps. Lag para velocidades por diff finita.
_SPEED_CAP_MPS  = 12.0          # cap anti-teleport del tracking


# ---- Adapter: frame de tracking PFF -> schema Z02 ----

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
    date_iso = str(md.get("date") or "")[:10]               # YYYY-MM-DD
    comp = md.get("competition") or {}
    comp_name = str(comp.get("name") if isinstance(comp, dict) else "")
    fps = float(md.get("fps") or 29.97)
    pitch_l = float(md.get("pitch_length") or PITCH_LENGTH)
    pitch_w = float(md.get("pitch_width") or PITCH_WIDTH)
    dt = vel_lag / fps                                       # delta_t entre frames cur y prev

    # Jerseys de portero por equipo (para colorear distinto en el render)
    gk: set[tuple[int, int]] = set()
    for r in load_rosters(match_id).iter_rows(named=True):
        if r["position_group"] == "GK" and r["shirt_number"] is not None:
            gk.add((int(r["team_id"]), int(r["shirt_number"])))

    # Pillo frame actual + el de hace vel_lag (para velocidades)
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
            px, py = prev_xy[side].get(j, (x, y))            # si no estaba antes, vel=0
            vx, vy = (x - px) / dt, (y - py) / dt
            sp = float(np.hypot(vx, vy))
            if sp > _SPEED_CAP_MPS:                           # capa teleports del tracking
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

    # Equipo atacante = POSESION REAL (game_event.home_ball), no heuristica de
    # posiciones. En frames de tracking home_ball viene null entre eventos, asi
    # que tomamos el valor no-null mas cercano (la posesion vigente del balon).
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
    # eventos, el frame de tracking esta rotado -> lo des-roto (x,y -> -x,-y) para
    # que todo viva en el frame de eventos. Auto-detectado por partido/periodo.
    ev_dir = attacking_direction(match_id).filter(
        (pl.col("team_id") == att_team_id) & (pl.col("period") == period))
    ev_right = (str(ev_dir["direction"][0]) == "R") if ev_dir.height else True
    tr_right = _attacking_right_tracking(match_id, att_team_id, period, home_id)
    if ev_right != tr_right:
        # PFF re-origina el frame de tracking en prorroga: rotacion 180 (x,y -> -x,-y)
        # respecto al frame de eventos. Verificado por jugador (corr_X y corr_Y < 0 en ET).
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
    Robusto al espejo eventos/tracking que PFF mete en prorroga en algunos
    partidos, porque mide sobre el mismo frame que se pinta.
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


# ---- Malla PPCF ----

def compute_ppcf_grid(frame_df: pd.DataFrame, att_team_id: int,
                       ball_pos: np.ndarray, pitch_l: float, pitch_w: float,
                       n_x: int = 80, n_y: int = 52) -> np.ndarray:
    """PPCF del equipo atacante en una malla n_y x n_x del campo (via Z02)."""
    xs = np.linspace(-pitch_l / 2, pitch_l / 2, n_x)
    ys = np.linspace(-pitch_w / 2, pitch_w / 2, n_y)
    XX, YY = np.meshgrid(xs, ys)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    ppcf = pc.ppcf_at_targets(frame_df, targets, att_team_id, ball_pos)
    return ppcf.reshape(n_y, n_x)


# ---- Layout pitch-aligned (header + pitch + footer al mismo ancho) ----
# Campo aspect = 105/68 = 1.544. Anclo header/footer al ancho exacto del pitch.

# ---- Layout: figura se ajusta al campo (sin margenes laterales) ----
# La figura tiene exactamente el ancho del campo. El alto se deriva para que
# el axes del pitch tenga el ratio correcto (datos x/y, no fisico 105/68):
#   xlim = [-55.5, 55.5] -> rango 111 m
#   ylim = [-37,   37  ] -> rango 74 m
#   ratio datos = 111/74 = 1.5 (ligeramente mas cuadrado que 105/68=1.544)
# Con axes width=FIG_W y height=FIG_W/1.5, set_aspect("equal") encaja perfecto.

FIG_W            = 14.0    # ancho fijo en pulgadas (= ancho fisico del campo)
HEADER_H_IN      = 1.1     # ↑ → header mas alto / ↓ → mas espacio al campo
FOOTER_H_IN      = 1.3     # ↑ → footer mas alto (leyenda)
PAD_TOP_IN       = 0.04    # margen entre tope de fig y header
PAD_HDR_PITCH_IN = 0.0     # separacion header-campo (0 = pegados)
PAD_PCH_FTR_IN   = 0.05    # separacion campo-footer
PAD_BOT_IN       = 0.07    # margen entre footer y base

_DATA_RATIO = (PITCH_LENGTH + 6.0) / (PITCH_WIDTH + 6.0)   # 111/74 ≈ 1.5
PITCH_H_IN  = FIG_W / _DATA_RATIO                           # alto del axes del campo

FIG_H = (PAD_TOP_IN + HEADER_H_IN + PAD_HDR_PITCH_IN
         + PITCH_H_IN + PAD_PCH_FTR_IN
         + FOOTER_H_IN + PAD_BOT_IN)

FIGSIZE = (FIG_W, FIG_H)


def _layout(figsize: tuple = FIGSIZE) -> dict:
    """Posiciones absolutas de header/pitch/footer. Pitch ocupa todo el ancho."""
    _, fh = figsize
    h_hdr = HEADER_H_IN      / fh
    h_pch = PITCH_H_IN       / fh
    h_ftr = FOOTER_H_IN      / fh
    p_top = PAD_TOP_IN       / fh
    p_hp  = PAD_HDR_PITCH_IN / fh
    p_pf  = PAD_PCH_FTR_IN   / fh
    p_bot = PAD_BOT_IN       / fh

    y_ftr_bot = p_bot
    y_pch_bot = y_ftr_bot + h_ftr + p_pf
    y_pch_top = y_pch_bot + h_pch
    y_hdr_bot = y_pch_top + p_hp
    return {
        "left": 0.0, "width": 1.0,
        "header": (0.0, y_hdr_bot, 1.0, h_hdr),
        "pitch":  (0.0, y_pch_bot, 1.0, h_pch),
        "footer": (0.0, y_ftr_bot, 1.0, h_ftr),
    }


def _make_block_axes(fig, left, bottom, width, height):
    """Sub-axes limpio (sin spines, lim 0-1). Para los bloques del footer."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax



def _draw_football_on_pitch(ax: plt.Axes, cx: float, cy: float,
                             r: float = 0.65, zorder: int = 10) -> None:
    """Balon de futbol (patron pentagono Telstar) en coords del campo.

    El campo debe tener set_aspect('equal') para que el circulo no se deforme.
    """
    n = 5
    ball = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor="black",
                            linewidth=1.5, zorder=zorder)
    ax.add_patch(ball)
    rp = r * 0.37
    c_ang = np.array([np.pi / 2 + 2 * np.pi * i / n for i in range(n)])
    pts_c = np.column_stack([cx + rp * np.cos(c_ang), cy + rp * np.sin(c_ang)])
    ax.add_patch(mpatches.Polygon(pts_c, facecolor="black", edgecolor="none",
                                   zorder=zorder + 1))
    r_outer, rp2 = r * 0.73, r * 0.27
    for i in range(n):
        ang = c_ang[i]
        ox = cx + r_outer * np.cos(ang)
        oy = cy + r_outer * np.sin(ang)
        o_ang = np.array([ang + 2 * np.pi * k / n for k in range(n)])
        pts_o = np.column_stack([ox + rp2 * np.cos(o_ang), oy + rp2 * np.sin(o_ang)])
        patch = mpatches.Polygon(pts_o, facecolor="black", edgecolor="none",
                                  zorder=zorder + 1)
        patch.set_clip_path(ball)
        ax.add_patch(patch)


# ---- Footer (3 bloques: leyenda + direccion + stats) ----

def _draw_footer(fig: plt.Figure, L: dict, att_team_name: str,
                  def_team_name: str, attacking_right: bool = True) -> None:
    """Footer 3 bloques 40/20/40 del ancho del pitch:
      block 1 (40%) — barra gradiente PPCF + etiquetas de control
      block 2 (20%) — triangulos direccion del juego
      block 3 (40%) — leyenda de nodos por equipo (azul atacante / rojo defensor)
    """
    fl = L["left"]; fw = L["width"]
    fb = L["footer"][1]; fh = L["footer"][3]
    b1_w, b2_w, b3_w = fw * 0.40, fw * 0.20, fw * 0.40    # split 40/20/40 del ancho del pitch
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ---- BLOCK 1: barra gradiente horizontal del PPCF + etiquetas ----
    bar_y  = 0.75                                      # ↑ barra SUBE / ↓ BAJA dentro del block
    bar_h  = 0.20                                         # ↑ barra MAS ALTA
    bar_x0 = 0.10                                         # ↑ extremo izq MAS a la DERECHA
    bar_x1 = 0.90                                         # ↑ extremo dcha MAS a la DERECHA
    n_steps = 200                                          # mas pasos = gradiente mas suave
    xs = np.linspace(bar_x0, bar_x1, n_steps + 1)
    for i in range(n_steps):
        c = PPCF_CMAP(i / (n_steps - 1))
        ax_b1.add_patch(mpatches.Rectangle(
            (xs[i], bar_y - bar_h / 2), xs[i + 1] - xs[i], bar_h,
            facecolor=c, edgecolor="none", transform=ax_b1.transAxes, zorder=2))
    # Borde fino blanco alrededor de la barra
    ax_b1.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y - bar_h / 2), bar_x1 - bar_x0, bar_h,
        facecolor="none", edgecolor=WHITE, lw=0.7,
        transform=ax_b1.transAxes, zorder=3))
    # Etiquetas extremos debajo de la barra (sin "neutro 50/50")
    ax_b1.text(bar_x0, bar_y - bar_h / 2 - 0.1, "100% defensor",
                ha="left", va="top", fontsize=12, color=WHITE,
                transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(bar_x1, bar_y - bar_h / 2 - 0.1, "100% atacante",
                ha="right", va="top", fontsize=12, color=WHITE,
                transform=ax_b1.transAxes, fontweight="bold")
    # Descripcion arriba (normal: ni bold ni cursiva), a la altura del antiguo titulo
    ax_b1.text(0.5, bar_y + bar_h / 2 + 0.1,
                "probabilidad de que el equipo controle ese punto del campo",
                ha="center", va="bottom", fontsize=10, color=WHITE,
                transform=ax_b1.transAxes)

    # ---- BLOCK 2: triangulos direccion del juego ----
    n_tri   = 4                                           # ↑ MAS triangulos / ↓ menos
    tri_w   = 0.10                                        # ↑ triangulo MAS ANCHO
    tri_h   = 0.16                                        # ↑ triangulo MAS ALTO
    spacing = 0.06                                        # ↑ MAS hueco entre triangulos
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0                       # centra los triangulos en el bloque
    y_tri   = 0.76                                        # ↑ triangulos SUBEN
    for i in range(n_tri):
        cx = start_x + i * (tri_w + spacing)
        if attacking_right:                               # triangulos apuntando a la DERECHA
            verts = [(cx, y_tri + tri_h / 2),
                      (cx + tri_w, y_tri),
                      (cx, y_tri - tri_h / 2)]
        else:                                             # triangulos apuntando a la IZQUIERDA
            verts = [(cx + tri_w, y_tri + tri_h / 2),
                      (cx, y_tri),
                      (cx + tri_w, y_tri - tri_h / 2)]
        alpha = 0.32 + i * 0.18                           # ↑ base/step -> triangulos MAS OPACOS
        ax_b2.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=ATT, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10))
    # Etiqueta unica (bold, sin cursiva)
    ax_b2.text(0.5, 0.5, "Dirección de Ataque",
                ha="center", va="center", fontsize=12, color=WHITE,
                transform=ax_b2.transAxes, fontweight="bold")

    # ---- BLOCK 3: equipos (izq + centro-izq) + balon (dcha) ----
    y_node = 0.75
    for x_node, x_txt, color, name in (
            (0.07, 0.145, ATT, att_team_name),            # IZQUIERDA: atacante
            (0.38, 0.455, DEF, def_team_name)):           # CENTRO: defensor
        ax_b3.plot(x_node, y_node, "o", ms=25, color=color, markeredgecolor=WHITE,
                    markeredgewidth=1.3, alpha=0.93, transform=ax_b3.transAxes,
                    clip_on=False, zorder=5)
        ax_b3.text(x_txt, y_node, name, ha="left", va="center", fontsize=12,
                    color=WHITE, fontweight="bold", transform=ax_b3.transAxes)
    # Balon al final (dcha): mismo patron Telstar que en el campo, compensado
    # por el aspect del footer (5.6"x1.3" = no-cuadrado) usando una Ellipse.
    x_ball = 0.70
    _b3_aspect = (FIG_W * b3_w) / FOOTER_H_IN          # ≈ 5.6 / 1.3 ≈ 4.31
    # ms=25pt en los nodos de equipo → diametro 25pt = 0.347" = 0.267 frac alto
    # → radio y = 0.134, mismo tamaño visible que los nodos France/Argentina
    _r_y = 0.134
    _r_x = _r_y / _b3_aspect                            # compensa pa que parezca circulo
    _n = 5
    _c_ang = np.array([np.pi / 2 + 2 * np.pi * i / _n for i in range(_n)])
    _ball_ell = mpatches.Ellipse((x_ball, y_node), 2 * _r_x, 2 * _r_y,
                                   facecolor="white", edgecolor="black",
                                   linewidth=1.5, zorder=5)
    ax_b3.add_patch(_ball_ell)
    # pentagono central negro
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
    ax_b3.text(x_ball + 0.075, y_node, "Balón", ha="left", va="center",
                fontsize=12, color=WHITE, fontweight="bold",
                transform=ax_b3.transAxes)


# ---- Render principal ----

def plot_ppcf(match_id: int, frame_num: int, title: Optional[str] = None,
               subtitle=None, save_path=None,
               attacking_right: Optional[bool] = None) -> plt.Figure:
    """Render del frame: header + superficie PPCF + jugadores + footer.

    Si title/subtitle son None, se generan automaticamente desde metadata.
    """
    df, ball_pos, att, meta = _load_frame_z02(match_id, frame_num)
    grid = compute_ppcf_grid(df, att, ball_pos, meta["pitch_l"], meta["pitch_w"])

    # Nombres atacante / defensor (resueltos desde el equipo del jug mas cercano al balon)
    att_team_name = meta["home_name"] if att == meta["home_id"] else meta["away_name"]
    def_team_name = meta["away_name"] if att == meta["home_id"] else meta["home_name"]
    att_slug = _TEAM_TO_SLUG.get(att_team_name)
    team_logo = (_LOGOS / f"{att_slug}.png") if att_slug else None

    # Direccion de ataque desde attacking_direction (frame de eventos = verdad
    # fisica). El tracking ya se des-espeja en _load_frame_z02 para vivir en este
    # mismo frame, asi que flecha y posiciones casan. 'R' -> ataca a la derecha.
    if attacking_right is None:
        d = attacking_direction(match_id).filter(
            (pl.col("team_id") == att) & (pl.col("period") == meta["period"]))
        attacking_right = (str(d["direction"][0]) == "R") if d.height else True

    # Marcador final del partido (subsubtitulo del header)
    g = goals_timeline(match_id)
    score_home = int(g["cum_home"].max() or 0) if g.height else 0
    score_away = int(g["cum_away"].max() or 0) if g.height else 0

    # Titulo + subtitulo auto-generados desde metadata (formato Opta-style)
    home_name = meta["home_name"]; away_name = meta["away_name"]
    if title is None:
        title = f"Pitch Control: {home_name} vs {away_name}"
    if subtitle is None:
        from datetime import datetime
        comp = meta["comp_name"] or "Mundial Qatar 2022"
        _MES = ["enero","febrero","marzo","abril","mayo","junio",
                "julio","agosto","septiembre","octubre","noviembre","diciembre"]
        try:
            dt = datetime.strptime(meta["date_iso"], "%Y-%m-%d")
            date_fmt = f"{dt.day} de {_MES[dt.month-1]} de {dt.year}"
        except Exception:
            date_fmt = meta["date_iso"]
        subtitle = (f"{comp}  |  "
                    f"{home_name} {score_home} - {score_away} {away_name}"
                    f"  ({date_fmt})")

    # ---- Construye fig + 2 axes (pitch / footer) ----
    L = _layout(FIGSIZE)
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)

    draw_header(fig, title=title,
                subtitle=subtitle if isinstance(subtitle, str) else None,
                escudo_path=(str(team_logo) if team_logo and team_logo.exists() else None),
                hdr_band=(L["header"][1], L["header"][1] + L["header"][3]),
                sub_size=13)

    ax_pitch = fig.add_axes(L["pitch"])
    draw_pitch(ax_pitch, meta["pitch_l"], meta["pitch_w"])
    L_h, W_h = meta["pitch_l"] / 2, meta["pitch_w"] / 2

    # ---- Superficie PPCF sobre el campo ----
    ax_pitch.imshow(grid, extent=[-L_h, L_h, -W_h, W_h], origin="lower",
                     cmap=PPCF_CMAP, vmin=0, vmax=1, alpha=0.72,
                     interpolation="spline36", zorder=1, aspect="auto")

    field = df[df["is_ball"] == 0]

    # ---- Flechas de velocidad por equipo ----
    # ↓ umbral 0.3 m/s captura tambien jugadores lentos (vs 0.6 que ocultaba muchos)
    # ↑ scale -> flechas MAS PEQUENAS; ↓ scale -> MAS GRANDES
    # ↑ width -> trazo MAS GRUESO
    # ↑ alpha -> MAS OPACO (mas visible sobre el PPCF)
    for is_att, color in ((True, ATT), (False, DEF)):
        sub = field[(field["team_id"] == att) == is_att]
        sub = sub[np.hypot(sub["vx"], sub["vy"]) > 0.3]
        if len(sub):
            ax_pitch.quiver(sub["x_tracking"], sub["y_tracking"],
                              sub["vx"], sub["vy"], color=color, scale=110,
                              scale_units="width", width=0.005, headwidth=4.0,
                              headlength=4.5, headaxislength=4.0, alpha=0.85,
                              zorder=3)

    # ---- Jugadores (circulo grande + dorsal con contorno BG) ----
    # ↑ ms -> jugador MAS GRANDE
    # ↑ markeredgewidth -> borde blanco MAS GRUESO
    for _, p in field.iterrows():
        color = (GK if p["is_goalkeeper"]
                  else (ATT if p["team_id"] == att else DEF))
        ax_pitch.plot(p["x_tracking"], p["y_tracking"], "o", ms=22, color=color,
                       markeredgecolor=WHITE, markeredgewidth=1.3, alpha=0.93,
                       zorder=5)
        ax_pitch.text(p["x_tracking"], p["y_tracking"], str(int(p["jersey"])),
                       color=WHITE, fontsize=8.5, ha="center", va="center",
                       fontweight="bold", zorder=6, path_effects=PE_S)

    # ---- Balon (patron pentagono Telstar) ----
    _draw_football_on_pitch(ax_pitch, ball_pos[0], ball_pos[1])

    _draw_footer(fig, L,
                  att_team_name=att_team_name, def_team_name=def_team_name,
                  attacking_right=attacking_right)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches=None (NO 'tight') porque el layout esta absoluto en fig coords
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
    return fig


# ---- CLI ----

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
    # Periodo se infiere desde endPeriod1: si el minuto pedido entra en P1, P1; si no, P2.
    md = load_metadata(mid).row(0, named=True)
    end_p1 = float(md.get("endPeriod1") or 3000.0) / 60.0
    period = 1 if match_min <= end_p1 + 1 else 2
    fnum = frame_for_clock(mid, period=period, clock_s=match_min * 60.0)
    out = f"outputs/viz/ppcf_{slug}.png"
    print(f"[ppcf] match {mid}  min {match_min} (P{period})  frame = {fnum}")
    plot_ppcf(mid, fnum, save_path=out)
    print(f"OK -> {out}")
