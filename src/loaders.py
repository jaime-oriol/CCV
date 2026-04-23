"""
loaders - Wrappers para cargar las fuentes de datos publicas.

Dos fuentes:
  - Wyscout   : socceraction PublicWyscoutLoader. Open dataset 2017/18.
                5 ligas (PL, La Liga, Serie A, Bundesliga, Ligue 1)
                + WC 2018 + Euro 2016. ~1.825 partidos. Corpus base
                para training de VAEP/xT/WP.
  - StatsBomb : socceraction StatsBombLoader. Subset filtrado a 4 torneos
                masculinos modernos con freeze-frames 360 (200 partidos).
                Para training de PSxG (near-miss) y cross-validacion
                de etiquetas vs PFF en el WC22.

PFF FC WC22 (target del TFM) va en un loader separado.
"""

from __future__ import annotations

from pathlib import Path

from socceraction.data.statsbomb import StatsBombLoader
from socceraction.data.wyscout import PublicWyscoutLoader

# -- Rutas ------------------------------------------------------------------

_PUBLIC    = Path(__file__).resolve().parents[1] / "data" / "public"
_WYSCOUT   = _PUBLIC / "wyscout"
_STATSBOMB = _PUBLIC / "statsbomb" / "data"

# -- Constantes -------------------------------------------------------------

# Wyscout: nombre de competicion -> (competition_id, season_id) para socceraction
WYSCOUT_COMPETITIONS: dict[str, tuple[int, int]] = {
    "England":               (364, 181150),
    "France":                (412, 181189),
    "Germany":               (426, 181137),
    "Italy":                 (524, 181248),
    "Spain":                 (795, 181144),
    "European_Championship": (102,   9291),
    "World_Cup":             ( 28,  10078),
}

# StatsBomb: nombre -> (competition_id, season_id). Solo lo que ha sobrevivido al filtro.
STATSBOMB_COMPETITIONS: dict[str, tuple[int, int]] = {
    "FIFA_World_Cup_2022":  (43, 106),    # 64 partidos, sagrado: solo cross-validacion
    "UEFA_Euro_2020":       (55,  43),    # 51 partidos, training PSxG
    "UEFA_Euro_2024":       (55, 282),    # 51 partidos, training PSxG
    "Bundesliga_2023_24":   ( 9, 281),    # 34 partidos, training PSxG
}


# -- Wyscout ----------------------------------------------------------------

def get_wyscout_loader() -> PublicWyscoutLoader:
    """Devuelve socceraction PublicWyscoutLoader apuntando a los datos locales.

    Uso:
        loader = get_wyscout_loader()
        comps  = loader.competitions()
        games  = loader.games(competition_id=364, season_id=181150)
        events = loader.events(game_id=games.game_id.iloc[0])

        # Convertir a acciones SPADL:
        from socceraction.spadl.wyscout import convert_to_actions
        actions = convert_to_actions(events, home_team_id=games.home_team_id.iloc[0])
    """
    return PublicWyscoutLoader(root=str(_WYSCOUT))


# -- StatsBomb --------------------------------------------------------------

def get_statsbomb_loader() -> StatsBombLoader:
    """Devuelve socceraction StatsBombLoader apuntando a los datos locales.

    Solo cubre los 4 torneos del subset filtrado (ver STATSBOMB_COMPETITIONS).

    Uso:
        loader  = get_statsbomb_loader()
        games   = loader.games(competition_id=55, season_id=43)  # Euro 2020
        events  = loader.events(game_id=games.game_id.iloc[0])
        teams   = loader.teams(game_id=games.game_id.iloc[0])
        players = loader.players(game_id=games.game_id.iloc[0])
    """
    return StatsBombLoader(root=str(_STATSBOMB), getter="local")
