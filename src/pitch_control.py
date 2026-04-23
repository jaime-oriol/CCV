"""
pitch_control - Pitch control (Spearman PPCF) from tracking data.

Adapted from LaurieOnTracking (Friends-of-Tracking-Data-FoTD) to work
with Opta Vision long-format tracking data (25fps, meters).

Core PPCF uses vectorized numpy (all grid targets computed simultaneously
via broadcasting). Mathematically identical to Spearman 2018 Eq 2-4.

Coordinate convention:
    Events/Runs: team-relative 0-100 (x=0 own goal, x=100 opp goal)
    Tracking: absolute 0-100 (fixed to pitch, direction_of_play per team)
    PPCF: meters centered at (0,0)
    Conversion: event_to_tracking() handles team-relative → absolute

Reference: Spearman 2018 "Beyond Expected Goals"

Functions:
    default_model_params       - PPCF model parameters
    opta_to_meters             - Absolute 0-100 → meters centered at (0,0)
    event_to_tracking          - Team-relative 0-100 → absolute 0-100
    compute_velocities         - Savitzky-Golay smoothed velocities
    compute_ppcf               - PPCF surface for a single frame (full grid)
    ppcf_at_targets            - PPCF at specific positions (uses same vectorized core)
    get_frame_at_event         - Closest tracking frame for an event timestamp
    compute_ppcf_at_event      - PPCF surface at event moment
    compute_delta_ppcf         - Delta PPCF between run start/end (for viz)
    compute_run_space_created  - m² of space gained by a run
    compute_indirect_attribution - INDIRECT detection via counterfactual PPCF at t=0
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# CONSTANTS ----------------------------------------------------------

PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters

# Event timeMin/timeSec are absolute from match start.
# Tracking uses timeelapsed_from_period_start_s (period-relative).
_PERIOD_OFFSETS = {1: 0, 2: 2700, 3: 5400, 4: 6300}


# COORDINATE CONVERSION ----------------------------------------------

def opta_to_meters(
    x, y,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
):
    """Convert absolute 0-100 coordinates to meters centered at pitch center.

    Tracking absolute: x=0 left, x=100 right, y=0 bottom, y=100 top.
    Meters: (0,0) = center, x in [-52.5, 52.5], y in [-34, 34].

    Works with scalars, arrays, or Series.
    """
    x_m = (x / 100.0) * pitch_length - pitch_length / 2.0
    y_m = (y / 100.0) * pitch_width - pitch_width / 2.0
    return x_m, y_m


def event_to_tracking(x, y, direction: str):
    """Convert team-relative 0-100 to absolute 0-100 (tracking space).

    Events/Runs use team-relative: x=0 own goal, x=100 opposition goal.
    Tracking uses absolute coordinates fixed to the pitch.

    Args:
        x, y: Team-relative positions (scalar, array, or Series).
        direction: "Left to Right" or "Right to Left" from tracking data.

    Returns:
        (x_abs, y_abs) in absolute 0-100.
    """
    if direction == "Left to Right":
        return x, y
    return 100.0 - x, 100.0 - y


# MODEL PARAMETERS ---------------------------------------------------

def default_model_params(time_to_control_veto: int = 3) -> dict:
    """Return default PPCF model parameters (Spearman 2018).

    Args:
        time_to_control_veto: Ignore players with control probability < 10^-veto.

    Returns:
        Dict with all model parameters.
    """
    params = {
        "max_player_speed": 5.0,       # m/s
        "reaction_time": 0.7,          # seconds
        "tti_sigma": 0.45,             # uncertainty in arrival time (s)
        "kappa_def": 1.0,              # defending advantage factor
        "lambda_att": 4.3,             # attacking ball control rate
        "average_ball_speed": 15.0,    # m/s
        "int_dt": 0.04,               # integration timestep (s)
        "max_int_time": 10.0,          # max integration time (s)
        "model_converge_tol": 0.01,    # convergence at PPCF > 0.99
    }
    params["lambda_def"] = params["lambda_att"] * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * 3.0

    sigma_term = np.sqrt(3) * params["tti_sigma"] / np.pi
    params["time_to_control_att"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_def"])
    )
    return params


# VELOCITY COMPUTATION -----------------------------------------------

def compute_velocities(
    tracking_df: pd.DataFrame,
    window: int = 7,
    polyorder: int = 1,
    max_speed: float = 12.0,
) -> pd.DataFrame:
    """Add vx, vy, speed columns using Savitzky-Golay filter.

    Computes smoothed velocities from x_tracking/y_tracking per player
    per period at 25fps (dt=0.04s). Caps outliers at max_speed.

    Args:
        tracking_df: Output of load_tracking.
        window: Savgol window size in frames (odd number).
        polyorder: Polynomial order for Savgol filter.
        max_speed: Cap speed at this value (m/s). Outliers set to NaN.

    Returns:
        tracking_df with added columns: vx, vy, speed.
    """
    df = tracking_df.copy()
    df["vx"] = np.nan
    df["vy"] = np.nan
    dt = 1.0 / 25.0  # 25fps constant

    for (pid, period), group in df.groupby(["player_id", "period_id"]):
        if pd.isna(pid):
            continue
        idx = group.index
        vx = group["x_tracking"].diff() / dt
        vy = group["y_tracking"].diff() / dt

        # Remove outliers exceeding max speed
        raw_speed = np.sqrt(vx**2 + vy**2)
        vx[raw_speed > max_speed] = np.nan
        vy[raw_speed > max_speed] = np.nan

        # Savitzky-Golay smoothing (need at least window+1 points)
        if len(vx.dropna()) > window:
            vx_smooth = savgol_filter(vx.fillna(0).values, window, polyorder)
            vy_smooth = savgol_filter(vy.fillna(0).values, window, polyorder)
            df.loc[idx, "vx"] = vx_smooth
            df.loc[idx, "vy"] = vy_smooth

    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    return df


# CORE PPCF ALGORITHM (Spearman 2018 / LaurieOnTracking) ------------
# Vectorized: all targets computed simultaneously via numpy broadcasting.
# Mathematically identical to per-target Euler integration (Eq 2-4).

def _ppcf_vectorized(
    targets: np.ndarray,
    att_pos: np.ndarray, att_vel: np.ndarray, att_gk: np.ndarray,
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray,
    ball_pos: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Vectorized PPCF for N targets simultaneously (Spearman 2018).

    Same Euler forward integration as the original per-target algorithm,
    but processes all grid/target positions in one pass using numpy
    broadcasting. Uses tau = t - ball_tt as integration variable so each
    target's integration is aligned to its own ball arrival time.

    Args:
        targets: Target positions in meters, shape (N, 2).
        att_pos, att_vel: Attacker positions/velocities, shape (Pa, 2).
        att_gk: Attacker goalkeeper mask, shape (Pa,).
        def_pos, def_vel: Defender positions/velocities, shape (Pd, 2).
        def_gk: Defender goalkeeper mask, shape (Pd,).
        ball_pos: Ball position in meters, shape (2,).
        params: PPCF model parameters.

    Returns:
        PPCF_att values, shape (N,).
    """
    N = len(targets)
    vmax = params["max_player_speed"]
    rt = params["reaction_time"]
    sigma = params["tti_sigma"]
    lam_a = params["lambda_att"]
    lam_d = params["lambda_def"]
    lam_gk = params["lambda_gk"]
    dt = params["int_dt"]
    max_int = params["max_int_time"]
    tol = params["model_converge_tol"]
    tc_a = params["time_to_control_att"]
    tc_d = params["time_to_control_def"]
    sig_c = np.pi / np.sqrt(3.0) / sigma
    Pa, Pd = len(att_pos), len(def_pos)

    # Ball travel time per target: (N,)
    if ball_pos is not None and not np.any(np.isnan(ball_pos)):
        ball_tt = np.linalg.norm(targets - ball_pos, axis=1) / params["average_ball_speed"]
    else:
        ball_tt = np.zeros(N)

    # Time-to-intercept: players × targets → (P, N)
    att_r = att_pos + att_vel * rt                                     # (Pa, 2)
    att_tti = rt + np.linalg.norm(
        targets[None] - att_r[:, None], axis=2,
    ) / vmax                                                           # (Pa, N)
    def_r = def_pos + def_vel * rt                                     # (Pd, 2)
    def_tti = rt + np.linalg.norm(
        targets[None] - def_r[:, None], axis=2,
    ) / vmax                                                           # (Pd, N)

    att_min = att_tti.min(axis=0) if Pa else np.full(N, np.inf)        # (N,)
    def_min = def_tti.min(axis=0) if Pd else np.full(N, np.inf)        # (N,)

    # Shortcuts: one team dominates → skip Euler (Spearman veto)
    result = np.full(N, np.nan)
    def_dom = (att_min - np.maximum(ball_tt, def_min)) >= tc_d
    att_dom = (def_min - np.maximum(ball_tt, att_min)) >= tc_a
    result[def_dom] = 0.0
    result[att_dom & ~def_dom] = 1.0

    eidx = np.where(np.isnan(result))[0]
    if len(eidx) == 0:
        return result

    # --- Euler integration for contested targets ---
    M = len(eidx)
    e_btt = ball_tt[eidx]                                              # (M,)
    e_att = att_tti[:, eidx]                                           # (Pa, M)
    e_def = def_tti[:, eidx]                                           # (Pd, M)

    # Active player masks per target: (P, M)
    a_act = (e_att - att_min[eidx]) < tc_a
    d_act = (e_def - def_min[eidx]) < tc_d

    # Defender lambda (GK gets higher rate): (Pd, 1)
    d_lam = np.where(def_gk, lam_gk, lam_d)[:, None]

    # Integrate in tau = t - ball_tt (aligned per target)
    tau_arr = np.arange(0, max_int, dt)
    att_tti_rel = e_att - e_btt[None, :]                               # (Pa, M)
    def_tti_rel = e_def - e_btt[None, :]                               # (Pd, M)

    pa_cum = np.zeros((Pa, M))
    pd_cum = np.zeros((Pd, M))
    tot_a = np.zeros(M)
    tot_d = np.zeros(M)
    conv = np.zeros(M, dtype=bool)

    for tau in tau_arr:
        if conv.all():
            break
        live = ~conv                                                   # (M,)
        rem = 1.0 - tot_a - tot_d                                     # (M,)
        with np.errstate(over='ignore'):
            p_a = 1.0 / (1.0 + np.exp(-sig_c * (tau - att_tti_rel)))  # (Pa, M)
            p_d = 1.0 / (1.0 + np.exp(-sig_c * (tau - def_tti_rel)))  # (Pd, M)
        pa_cum += (rem * p_a * lam_a * a_act * live) * dt
        pd_cum += (rem * p_d * d_lam * d_act * live) * dt
        tot_a = pa_cum.sum(axis=0)
        tot_d = pd_cum.sum(axis=0)
        conv = (tot_a + tot_d) > (1.0 - tol)

    result[eidx] = tot_a
    return result


# INTERNAL HELPERS ---------------------------------------------------

def _extract_teams(frame_data: pd.DataFrame, att_team_id):
    """Extract (pos, vel, is_gk) arrays for attacking and defending teams."""
    players = frame_data[frame_data["is_ball"] == 0]
    att = players[players["team_id"] == att_team_id]
    def_ = players[players["team_id"] != att_team_id]

    def _arrays(df):
        pos = df[["x_tracking", "y_tracking"]].values
        if "vx" in df.columns and "vy" in df.columns:
            vel = df[["vx", "vy"]].fillna(0).values
        else:
            vel = np.zeros_like(pos)
        is_gk = df["is_goalkeeper"].values.astype(bool)
        return pos, vel, is_gk

    return _arrays(att), _arrays(def_)


def _get_ball_pos(frame_data: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract ball position in meters from a tracking frame."""
    ball = frame_data[frame_data["is_ball"] == 1]
    if ball.empty:
        return None
    return np.array([ball.iloc[0]["x_tracking"], ball.iloc[0]["y_tracking"]])


# PPCF SURFACE -------------------------------------------------------

def compute_ppcf(
    frame_data: pd.DataFrame,
    att_team_id,
    ball_pos: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
    field_dimen: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PPCF surface for a single tracking frame.

    Args:
        frame_data: Tracking rows for one frame (all players + ball).
        att_team_id: Team ID of the attacking (possessing) team.
        ball_pos: Ball position in meters [x, y]. Auto-detected if None.
        params: Model parameters (default if None).
        n_grid_x: Grid resolution in x direction.
        field_dimen: Pitch dimensions in meters (length, width).

    Returns:
        (ppcf_att, xgrid, ygrid) where ppcf_att is shape (n_grid_y, n_grid_x).
    """
    if params is None:
        params = default_model_params()

    n_grid_y = int(n_grid_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_x
    dy = field_dimen[1] / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - field_dimen[1] / 2.0 + dy / 2.0

    if ball_pos is None:
        ball_pos = _get_ball_pos(frame_data)

    (att_pos, att_vel, att_gk), (def_pos, def_vel, def_gk) = _extract_teams(
        frame_data, att_team_id
    )

    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    ppcf_att = _ppcf_vectorized(
        targets, att_pos, att_vel, att_gk,
        def_pos, def_vel, def_gk, ball_pos, params,
    ).reshape(n_grid_y, n_grid_x)

    return ppcf_att, xgrid, ygrid


# POINT-WISE PPCF ---------------------------------------------------

def ppcf_at_targets(
    frame_data: pd.DataFrame,
    targets_meters: np.ndarray,
    att_team_id,
    ball_pos: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Compute PPCF at specific target positions (no grid).

    Much faster than compute_ppcf when only a few positions are needed
    (e.g., pass option positions for indirect attribution).

    Args:
        frame_data: Tracking rows for one frame.
        targets_meters: Array of shape (N, 2) with positions in meters.
        att_team_id: Team ID of the attacking team.
        ball_pos: Ball position in meters [x, y]. Auto-detected if None.
        params: Model parameters (default if None).

    Returns:
        Array of shape (N,) with PPCF_att at each target.
    """
    if params is None:
        params = default_model_params()
    if ball_pos is None:
        ball_pos = _get_ball_pos(frame_data)

    (att_pos, att_vel, att_gk), (def_pos, def_vel, def_gk) = _extract_teams(
        frame_data, att_team_id
    )

    return _ppcf_vectorized(
        targets_meters, att_pos, att_vel, att_gk,
        def_pos, def_vel, def_gk, ball_pos, params,
    )


# EVENT HELPERS ------------------------------------------------------

def get_frame_at_event(
    tracking_df: pd.DataFrame,
    event_row: pd.Series,
) -> pd.DataFrame:
    """Get the tracking frame closest to an event's timestamp.

    Converts event absolute time (timeMin*60+timeSec) to period-relative
    for matching with tracking's timeelapsed_from_period_start_s.

    Args:
        tracking_df: Full tracking data.
        event_row: Single event row from load_events.

    Returns:
        DataFrame with all rows for the closest frame.
    """
    period = int(event_row["periodId"])
    event_time_s = event_row["timeMin"] * 60 + event_row["timeSec"]
    event_time_s -= _PERIOD_OFFSETS.get(period, 0)

    period_data = tracking_df[tracking_df["period_id"] == period]
    frame_times = period_data.groupby("frame_count")[
        "timeelapsed_from_period_start_s"
    ].first()
    closest_frame = (frame_times - event_time_s).abs().idxmin()
    return tracking_df[tracking_df["frame_count"] == closest_frame]



def compute_ppcf_at_event(
    tracking_df: pd.DataFrame,
    event_row: pd.Series,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PPCF surface at the moment of a specific event.

    Args:
        tracking_df: Full tracking data with velocities.
        event_row: Single event row from load_events.
        params: Model parameters.
        n_grid_x: Grid resolution.

    Returns:
        (ppcf_att, xgrid, ygrid).
    """
    frame_data = get_frame_at_event(tracking_df, event_row)
    ball_pos = _get_ball_pos(frame_data)
    att_team_id = event_row["contestantId"]
    return compute_ppcf(frame_data, att_team_id, ball_pos, params, n_grid_x)


# DELTA PPCF (SPACE CREATED BY RUNS) --------------------------------

def compute_delta_ppcf(
    tracking_df: pd.DataFrame,
    run_row: pd.Series,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
    end_frame_override: Optional[int] = None,
    end_ball_pos: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute delta PPCF between run start and a reference frame.

    Positive values = space gained by the attacking team.
    Used for visualization (2×2 pitch control layout).

    Args:
        tracking_df: Full tracking data with velocities.
        run_row: Single run row from load_runs.
        params: Model parameters.
        n_grid_x: Grid resolution.
        end_frame_override: Use this frame instead of run endFrame.
            For INDIRECT: pass the event frame (t=0) so the surface
            reflects the state at pass moment.
        end_ball_pos: Override ball position (meters, centered) for the
            end frame PPCF. Usually not needed when using event frame
            (ball is naturally at passer in tracking).

    Returns:
        (delta_ppcf, ppcf_start, xgrid, ygrid) or all None if frames missing.
    """
    start_frame = int(run_row["startFrame"])
    end_frame = end_frame_override if end_frame_override is not None else int(run_row["endFrame"])
    att_team = run_row["contestantId"]

    frame_start = tracking_df[tracking_df["frame_count"] == start_frame]
    frame_end = tracking_df[tracking_df["frame_count"] == end_frame]

    if frame_start.empty or frame_end.empty:
        return None, None, None, None

    ball_start = _get_ball_pos(frame_start)

    ppcf_start, xgrid, ygrid = compute_ppcf(
        frame_start, att_team, ball_start, params, n_grid_x
    )
    ball_end = end_ball_pos if end_ball_pos is not None else _get_ball_pos(frame_end)
    ppcf_end, _, _ = compute_ppcf(
        frame_end, att_team, ball_end, params, n_grid_x
    )

    return ppcf_end - ppcf_start, ppcf_start, xgrid, ygrid


def compute_run_space_created(
    delta_ppcf: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
) -> float:
    """Compute total m² of space gained from a delta PPCF surface.

    Integrates only positive delta values (space gained by attacking team).
    """
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    return float(np.sum(np.maximum(delta_ppcf, 0)) * dx * dy)


# INDIRECT ATTRIBUTION -----------------------------------------------

def _point_to_segment_dist(p, a, b):
    """Distance from point p to line segment a-b (all in meters)."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-10), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def compute_indirect_attribution(
    tracking_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    params: Optional[dict] = None,
    ppcf_threshold: float = 0.05,
    corridor_m: float = 10.0,
    min_drag_m: float = 2.0,
    min_direction_cos: float = 0.866,
) -> pd.DataFrame:
    """Detect INDIRECT run impact via counterfactual PPCF at t=0.

    Causal chain: run → defenders follow (drag) → vacate original positions
    → receiver benefits from reduced defensive presence. Isolated via
    counterfactual: PPCF at receiver with real positions vs PPCF with
    dragged defenders placed back at their pre-drag positions (velocity=0).

    All evaluation at t=0 (pass moment). The run's value is the space it
    created when the passer decided, not what happens during ball flight.

    Algorithm:
        1. Defenders near run corridor at run_start (within corridor_m)
        2. Exclude defenders identified as pressing the passer (MA36
           pressure.player) — pressing ≠ being dragged by a run
        3. Defender moved in run direction by t=0 (cosine >= min_direction_cos,
           displacement >= min_drag_m) → confirmed drag
        4. Counterfactual PPCF at receiver position:
           - PPCF_real = pitch control at t=0 (real positions)
           - PPCF_counterfactual = pitch control at t=0 with dragged defenders
             returned to run_start positions (velocity=0)
           - causal_delta = PPCF_real - PPCF_counterfactual
           If causal_delta > ppcf_threshold → INDIRECT.

    No arbitrary proximity threshold: the counterfactual naturally returns
    ~0 when dragged defenders were far from the receiver.
    Best run selected by highest causal_delta (isolated PPCF gain).

    Args:
        tracking_df: Tracking WITH velocities (call compute_velocities first).
        landscape_df: Output of link_runs_to_options.
        runs_df: MA58 runs.
        events_df: MA36 events.
        params: PPCF model parameters (default if None).
        ppcf_threshold: Min counterfactual PPCF delta for INDIRECT (0.05).
        corridor_m: Max distance (m) from run segment to consider defenders.
        min_drag_m: Min displacement (m) in run direction to count as drag.
        min_direction_cos: Min cosine between defender displacement and run
            direction. 0.866 = within ~30°.

    Returns:
        DataFrame with one row per candidate event:
            id, indirect_delta, causal_delta, indirect_ppcf_real,
            indirect_ppcf_counterfactual, indirect_flag, drag_score,
            n_dragged, vacated_dist, indirect_run_id.
    """
    if params is None:
        params = default_model_params()

    _COLS = ["id", "indirect_delta", "causal_delta",
             "indirect_ppcf_real", "indirect_ppcf_counterfactual",
             "indirect_flag", "drag_score", "n_dragged", "vacated_dist",
             "indirect_run_id"]

    # --- Candidate events: chosen is NOT runner, but event has runs ---
    chosen = landscape_df[landscape_df["is_chosen"] == 1]
    run_max = landscape_df.groupby("id")["has_run"].max()
    candidate_ids = chosen[
        (chosen["has_run"] == 0)
        & (chosen["id"].map(run_max).fillna(0).astype(int) == 1)
    ]["id"].unique()

    if len(candidate_ids) == 0:
        return pd.DataFrame(columns=_COLS)

    # --- Lookups (built once) ---
    ev = events_df.drop_duplicates(subset="id").set_index("id")

    dir_lookup = (
        tracking_df[tracking_df["direction_of_play"].notna()]
        .groupby(["team_id", "period_id"])["direction_of_play"]
        .first().to_dict()
    )

    period_frame_times = {}
    for pid in tracking_df["period_id"].unique():
        pt = tracking_df[tracking_df["period_id"] == pid]
        period_frame_times[int(pid)] = (
            pt.groupby("frame_count")["timeelapsed_from_period_start_s"].first()
        )

    run_idx = runs_df.set_index("id")

    results = []

    for eid in candidate_ids:
        if eid not in ev.index:
            continue
        event = ev.loc[eid]
        att_team = event["contestantId"]
        period = int(event["periodId"])
        direction = dir_lookup.get((float(att_team), period), "Left to Right")

        # --- Receiver position at t=0 (pass moment) ---
        ch = chosen[chosen["id"] == eid]
        if ch.empty:
            continue
        rec_x = float(ch.iloc[0]["positionX"])
        rec_y = float(ch.iloc[0]["positionY"])

        rx_abs, ry_abs = event_to_tracking(rec_x, rec_y, direction)
        rx_m, ry_m = opta_to_meters(rx_abs, ry_abs)
        target = np.array([[rx_m, ry_m]])

        # Pressers from MA36 (exclude from drag detection)
        _pp = event.get("pressure.player")
        if isinstance(_pp, list):
            presser_pids = {float(p["playerId"]) for p in _pp}
        else:
            presser_pids = set()

        # --- Fallback: reception point (where ball actually goes) ---
        target_fb = None
        _rec_x_col = "reception.receivingX"
        _rec_y_col = "reception.receivingY"
        if (hasattr(event, "__getitem__") and
                pd.notna(event.get(_rec_x_col)) and
                pd.notna(event.get(_rec_y_col))):
            fb_x, fb_y = event_to_tracking(
                float(event[_rec_x_col]), float(event[_rec_y_col]), direction,
            )
            fb_m_x, fb_m_y = opta_to_meters(fb_x, fb_y)
            target_fb = np.array([[fb_m_x, fb_m_y]])

        # --- Event frame (t=0 = pass moment) ---
        if period not in period_frame_times:
            continue
        ft = period_frame_times[period]
        event_abs_s = event["timeMin"] * 60 + event["timeSec"]
        event_rel_s = event_abs_s - _PERIOD_OFFSETS.get(period, 0)
        event_frame = int((ft - event_rel_s).abs().idxmin())

        # Evaluate at pass moment (t=0), not reception
        t2_frame = event_frame

        # --- Linked runs: evaluate each, pick best drag ---
        ev_runs = landscape_df[
            (landscape_df["id"] == eid) & (landscape_df["has_run"] == 1)
        ]["run_id"].dropna().unique()
        if len(ev_runs) == 0:
            continue

        period_trk = tracking_df[tracking_df["period_id"] == period]

        # --- Frame at t=0 (pass moment) + real PPCF (computed ONCE) ---
        frame_t2 = period_trk[period_trk["frame_count"] == t2_frame]
        if frame_t2.empty:
            continue
        ball_pos_t2 = _get_ball_pos(frame_t2)
        ppcf_real = float(ppcf_at_targets(
            frame_t2, target, att_team, ball_pos_t2, params
        )[0])
        ppcf_real_fb = None
        if target_fb is not None:
            ppcf_real_fb = float(ppcf_at_targets(
                frame_t2, target_fb, att_team, ball_pos_t2, params
            )[0])

        best_run_result = None

        for rid in ev_runs:
            rid_int = int(rid)
            if rid_int not in run_idx.index:
                continue
            run = run_idx.loc[rid_int]
            if pd.isna(run.get("startFrame")) or pd.isna(run.get("startX")):
                continue

            t1_frame = int(run["startFrame"])
            run_team = run["contestantId"]
            runner_pid = float(run["playerId"])

            # Run start in meters (from MA58)
            rs_abs, rs_y_abs = event_to_tracking(
                float(run["startX"]), float(run["startY"]), direction
            )
            run_start_m = np.array(opta_to_meters(rs_abs, rs_y_abs))

            # Run end = runner's ACTUAL tracking position at t2 (pass moment),
            # NOT MA58 endXY which may be beyond the pass moment.
            runner_at_t2 = period_trk[
                (period_trk["frame_count"] == t2_frame)
                & (period_trk["player_id"] == runner_pid)
            ]
            if not runner_at_t2.empty:
                run_end_m = np.array(opta_to_meters(
                    float(runner_at_t2.iloc[0]["x"]),
                    float(runner_at_t2.iloc[0]["y"]),
                ))
            else:
                # Fallback: MA58 endXY (run already finished before pass)
                re_abs, re_y_abs = event_to_tracking(
                    float(run["endX"]), float(run["endY"]), direction
                )
                run_end_m = np.array(opta_to_meters(re_abs, re_y_abs))

            run_dir = run_end_m - run_start_m
            if np.linalg.norm(run_dir) < 0.5:
                continue  # runner barely moved by pass moment
            run_dir_norm = run_dir / (np.linalg.norm(run_dir) + 1e-10)

            # --- Defenders at run_start frame ---
            frame_t1 = period_trk[period_trk["frame_count"] == t1_frame]
            if frame_t1.empty:
                continue
            defs_t1 = frame_t1[
                (frame_t1["team_id"] != float(run_team))
                & (frame_t1["is_ball"] != 1)
            ]
            if defs_t1.empty:
                continue

            # Filter defenders near run corridor
            def_pids = []
            def_start_pos_m = []  # meters (for distance checks)
            def_start_pos_100 = []  # tracking 0-100 (for counterfactual)
            for _, d in defs_t1.iterrows():
                dm = np.array(opta_to_meters(float(d["x"]), float(d["y"])))
                dist_to_seg = _point_to_segment_dist(dm, run_start_m, run_end_m)
                if dist_to_seg <= corridor_m:
                    def_pids.append(d["player_id"])
                    def_start_pos_m.append(dm)
                    def_start_pos_100.append((float(d["x"]), float(d["y"])))

            if not def_pids:
                continue

            # --- Same defenders at pass moment (t=0) ---
            drag_scores = []
            vacated_positions = []
            dragged_pids = []
            dragged_orig_100 = []  # tracking coords for counterfactual
            dragged_orig_m = []
            for pid_d, pos_start_m, pos_start_100 in zip(
                def_pids, def_start_pos_m, def_start_pos_100
            ):
                d_t2 = frame_t2[frame_t2["player_id"] == pid_d]
                if d_t2.empty:
                    continue
                pos_end = np.array(opta_to_meters(
                    float(d_t2.iloc[0]["x"]), float(d_t2.iloc[0]["y"])
                ))
                displacement = pos_end - pos_start_m
                disp_norm = np.linalg.norm(displacement)
                if disp_norm < 0.5:
                    continue  # barely moved
                # Direction check: defender moved in similar direction as run
                cos_sim = float(np.dot(displacement / disp_norm, run_dir_norm))
                if cos_sim < min_direction_cos:
                    continue  # moved in a different direction
                # Exclude defenders identified as pressing the passer
                if pid_d in presser_pids:
                    continue
                drag = float(np.dot(displacement, run_dir_norm))
                if drag >= min_drag_m:
                    drag_scores.append(drag)
                    vacated_positions.append(pos_start_m)
                    dragged_pids.append(pid_d)
                    dragged_orig_100.append(pos_start_100)
                    dragged_orig_m.append(pos_start_m)

            if not drag_scores:
                continue

            mean_drag = float(np.mean(drag_scores))

            # --- Counterfactual PPCF for THIS run's dragged defenders ---
            frame_cf = frame_t2.copy()
            for pid_d, (ox, oy), (ox_m, oy_m) in zip(
                dragged_pids, dragged_orig_100, dragged_orig_m,
            ):
                mask = frame_cf["player_id"] == pid_d
                if mask.any():
                    frame_cf.loc[mask, "x"] = float(ox)
                    frame_cf.loc[mask, "y"] = float(oy)
                    frame_cf.loc[mask, "x_tracking"] = np.float32(ox_m)
                    frame_cf.loc[mask, "y_tracking"] = np.float32(oy_m)
                    frame_cf.loc[mask, "vx"] = np.float32(0.0)
                    frame_cf.loc[mask, "vy"] = np.float32(0.0)

            ppcf_cf = float(ppcf_at_targets(
                frame_cf, target, att_team, ball_pos_t2, params
            )[0])
            causal_delta = ppcf_real - ppcf_cf

            # Fallback: if receiver fails, try reception point
            used_fb = False
            if causal_delta <= ppcf_threshold and target_fb is not None:
                ppcf_cf_fb = float(ppcf_at_targets(
                    frame_cf, target_fb, att_team, ball_pos_t2, params
                )[0])
                cd_ball = ppcf_real_fb - ppcf_cf_fb
                if cd_ball > causal_delta:
                    causal_delta = cd_ball
                    ppcf_cf = ppcf_cf_fb
                    used_fb = True

            # Keep best run (highest causal_delta = most isolated PPCF gain)
            if best_run_result is None or causal_delta > best_run_result["causal_delta"]:
                eval_m = (np.array([target_fb[0, 0], target_fb[0, 1]])
                          if used_fb else np.array([rx_m, ry_m]))
                vac_dists = [float(np.linalg.norm(eval_m - vp))
                             for vp in vacated_positions]
                best_run_result = {
                    "rid": rid_int,
                    "t1_frame": t1_frame,
                    "drag_score": round(mean_drag, 2),
                    "n_dragged": len(drag_scores),
                    "vacated_dist": round(min(vac_dists), 1),
                    "causal_delta": round(causal_delta, 4),
                    "ppcf_cf": round(ppcf_cf, 4),
                }

        if best_run_result is None:
            continue

        # Total delta (t1 → t2) for reference
        _ppcf_real_final = ppcf_real
        _target_final = target
        if target_fb is not None and ppcf_real_fb is not None:
            # Use whichever evaluation point gave the best causal_delta
            _cf_recv = ppcf_real - best_run_result["ppcf_cf"]
            if best_run_result["causal_delta"] > _cf_recv + 1e-6:
                _ppcf_real_final = ppcf_real_fb
                _target_final = target_fb

        frame_t1 = period_trk[
            period_trk["frame_count"] == best_run_result["t1_frame"]
        ]
        if not frame_t1.empty:
            ball_pos_t1 = _get_ball_pos(frame_t1)
            ppcf_t1 = float(ppcf_at_targets(
                frame_t1, _target_final, att_team, ball_pos_t1, params
            )[0])
            total_delta = _ppcf_real_final - ppcf_t1
        else:
            total_delta = best_run_result["causal_delta"]

        # INDIRECT = counterfactual causal delta above threshold
        is_indirect = int(best_run_result["causal_delta"] > ppcf_threshold)

        results.append({
            "id": eid,
            "indirect_delta": round(total_delta, 4),
            "causal_delta": best_run_result["causal_delta"],
            "indirect_ppcf_real": round(_ppcf_real_final, 4),
            "indirect_ppcf_counterfactual": best_run_result["ppcf_cf"],
            "indirect_flag": is_indirect,
            "drag_score": best_run_result["drag_score"],
            "n_dragged": best_run_result["n_dragged"],
            "vacated_dist": best_run_result["vacated_dist"],
            "indirect_run_id": best_run_result["rid"],
        })

    if not results:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(results)
