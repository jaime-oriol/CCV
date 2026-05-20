"""extract.assets - Descarga escudos selecciones + caras jugadores.

Logos: sportlogos/sport.db.logos (PNG transparente real, escudo individual
       de federacion, organizados por continente). 32 selecciones WC22.
Caras: FotMob CDN (PNG transparente). Search por nombre con fallback en
       guion -> espacio y apellido solo.

Outputs:
    outputs/assets/logos/{slug}.png         32 selecciones (iso3)
    outputs/assets/faces/{pff_player_id}.png
    outputs/assets/manifest.csv             mapping completo + status

Idempotente: cached salta. Manifest se fusiona (logos y faces se actualizan
por separado sin pisar el otro).

Uso:
    python -m src.extract.assets --logos    # solo escudos
    python -m src.extract.assets --faces    # solo caras (lee pcj_table)
    python -m src.extract.assets --all      # ambos
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

ROOT      = Path(__file__).resolve().parents[2]
OUT_LOGOS = ROOT / "outputs" / "assets" / "logos"
OUT_FACES = ROOT / "outputs" / "assets" / "faces"
MANIFEST  = ROOT / "outputs" / "assets" / "manifest.csv"
PCJ_TABLE = ROOT / "outputs" / "pcj_table.parquet"

# Endpoints externos
SPORTLOGOS_BASE = "https://raw.githubusercontent.com/sportlogos/sport.db.logos/master"
FOTMOB_SEARCH = "https://www.fotmob.com/api/data/search/suggest?hits=50&lang=en,de,pl,da&term={q}"
FOTMOB_IMG    = "https://images.fotmob.com/image_resources/playerimages/{id}.png"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fotmob.com/",
    "Accept-Encoding": "gzip",
}

# pcj.team_name -> (continent_folder, iso3_slug) en sportlogos/sport.db.logos
TEAM_TO_PATH = {
    "Argentina":     ("south-america", "arg"), "Brazil":      ("south-america", "bra"),
    "Ecuador":       ("south-america", "ecu"), "Uruguay":     ("south-america", "uru"),
    "Belgium":       ("europe", "bel"), "Croatia":     ("europe", "cro"),
    "Denmark":       ("europe", "den"), "England":     ("europe", "eng"),
    "France":        ("europe", "fra"), "Germany":     ("europe", "ger"),
    "Netherlands":   ("europe", "ned"), "Poland":      ("europe", "pol"),
    "Portugal":      ("europe", "por"), "Serbia":      ("europe", "srb"),
    "Spain":         ("europe", "esp"), "Switzerland": ("europe", "sui"),
    "Wales":         ("europe", "wal"),
    "Cameroon":      ("africa", "cmr"), "Ghana":       ("africa", "gha"),
    "Morocco":       ("africa", "mar"), "Senegal":     ("africa", "sen"),
    "Tunisia":       ("africa", "tun"),
    "Japan":         ("asia", "jpn"), "South Korea":   ("asia", "kor"),
    "Iran":          ("middle-east", "irn"), "Qatar":  ("middle-east", "qat"),
    "Saudi Arabia":  ("middle-east", "ksa"),
    "Canada":        ("north-america", "can"), "Mexico":("north-america", "mex"),
    "United States": ("north-america", "usa"),
    "Costa Rica":    ("central-america", "crc"),
    "Australia":     ("pacific", "aus"),
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    """GET con headers tipo browser + gzip transparente."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw


def fetch_logos(verbose: bool = True) -> list[dict]:
    """Descarga los 32 escudos WC22 a outputs/assets/logos/{slug}.png."""
    OUT_LOGOS.mkdir(parents=True, exist_ok=True)
    rows = []
    for team, (cont, slug) in TEAM_TO_PATH.items():
        dst = OUT_LOGOS / f"{slug}.png"
        if dst.exists():
            rows.append({"team_name": team, "slug": slug,
                          "file": str(dst.relative_to(ROOT)), "status": "cached"})
            if verbose:
                print(f"  cache  {team:18s} -> {dst.name}")
            continue
        url = f"{SPORTLOGOS_BASE}/{cont}/{slug}.png"
        try:
            raw = _http_get(url)
            dst.write_bytes(raw)
            rows.append({"team_name": team, "slug": slug,
                          "file": str(dst.relative_to(ROOT)), "status": "ok"})
            if verbose:
                print(f"  ok     {team:18s} -> {dst.name}  ({len(raw)//1024} KB)")
        except Exception as e:
            rows.append({"team_name": team, "slug": slug, "file": "",
                          "status": f"err:{e}"})
            if verbose:
                print(f"  ERR    {team:18s} {e}")
        time.sleep(0.15)         # rate-limit suave para no martillear GitHub
    return rows


def fotmob_search(term: str) -> list[dict]:
    """Busca jugadores en FotMob por nombre. Devuelve los hits tipo 'player'."""
    url = FOTMOB_SEARCH.format(q=urllib.parse.quote(term))
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        if isinstance(data, list) and data:
            return [s for s in data[0].get("suggestions", [])
                    if s.get("type") == "player"]
    except Exception:
        return []
    return []


def fotmob_pick(name: str, team: str) -> tuple[int | None, str]:
    """Resuelve nombre -> fotmob_id. Cascada: nombre exacto -> sin guion -> apellido."""
    variants = [name, name.replace("-", " "), name.split()[-1]]
    for v in dict.fromkeys(variants):                    # dedupe manteniendo orden
        hits = fotmob_search(v)
        if hits:
            return int(hits[0]["id"]), f"hit:{v}"
        time.sleep(0.25)
    return None, "no_hit"


def fetch_faces(verbose: bool = True) -> list[dict]:
    """Descarga las caras de TODOS los jugadores en outputs/pcj_table.parquet."""
    OUT_FACES.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(PCJ_TABLE).select(
        ["pff_player_id", "player_name", "team_name"]
    ).unique()
    rows = []
    n = df.height
    for i, r in enumerate(df.iter_rows(named=True), 1):
        pid = int(r["pff_player_id"])
        name = r["player_name"]
        team = r["team_name"]
        dst = OUT_FACES / f"{pid}.png"
        if dst.exists():
            rows.append({"pff_player_id": pid, "player_name": name, "team_name": team,
                          "fotmob_id": "", "file": str(dst.relative_to(ROOT)),
                          "status": "cached"})
            continue
        fid, status = fotmob_pick(name, team)
        if fid is None:
            rows.append({"pff_player_id": pid, "player_name": name, "team_name": team,
                          "fotmob_id": "", "file": "", "status": status})
            if verbose:
                print(f"  [{i:>3}/{n}] MISS  {name:30s} ({team})")
            continue
        try:
            raw = _http_get(FOTMOB_IMG.format(id=fid))
            dst.write_bytes(raw)
            rows.append({"pff_player_id": pid, "player_name": name, "team_name": team,
                          "fotmob_id": fid, "file": str(dst.relative_to(ROOT)),
                          "status": "ok"})
            if verbose:
                print(f"  [{i:>3}/{n}] ok    {name:30s} ({team})  fid={fid}"
                       f"  {len(raw)//1024} KB")
        except urllib.error.HTTPError as e:
            # 403 = FotMob no tiene foto subida para ese jugador
            rows.append({"pff_player_id": pid, "player_name": name, "team_name": team,
                          "fotmob_id": fid, "file": "", "status": f"http:{e.code}"})
            if verbose:
                print(f"  [{i:>3}/{n}] HTTP{e.code}  {name:30s} fid={fid}")
        except Exception as e:
            rows.append({"pff_player_id": pid, "player_name": name, "team_name": team,
                          "fotmob_id": fid, "file": "", "status": f"err:{e}"})
        time.sleep(0.3)
    return rows


def _read_manifest() -> list[dict]:
    """Lee manifest existente (o lista vacia si no existe)."""
    if not MANIFEST.exists():
        return []
    with MANIFEST.open() as f:
        return list(csv.DictReader(f))


def write_manifest(logos_rows: list[dict], faces_rows: list[dict]) -> None:
    """Manifest fusionado: actualiza solo el kind que se le pasa, conserva el otro."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest()
    keep = [r for r in existing
            if (r["kind"] == "logo" and not logos_rows)
            or (r["kind"] == "face" and not faces_rows)]
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "key", "name", "team", "fotmob_id", "file", "status"])
        for r in keep:
            w.writerow([r["kind"], r["key"], r["name"], r["team"],
                         r["fotmob_id"], r["file"], r["status"]])
        for r in logos_rows:
            w.writerow(["logo", r["slug"], "", r["team_name"], "",
                         r["file"], r["status"]])
        for r in faces_rows:
            w.writerow(["face", r["pff_player_id"], r["player_name"],
                         r["team_name"], r["fotmob_id"], r["file"], r["status"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logos", action="store_true", help="descarga escudos selecciones")
    ap.add_argument("--faces", action="store_true", help="descarga caras jugadores")
    ap.add_argument("--all", action="store_true", help="ambos")
    args = ap.parse_args()
    do_logos = args.logos or args.all
    do_faces = args.faces or args.all
    if not (do_logos or do_faces):
        ap.error("indica --logos, --faces, o --all")
    logos_rows, faces_rows = [], []
    if do_logos:
        print("---- LOGOS ----")
        logos_rows = fetch_logos()
        ok = sum(1 for r in logos_rows if r["status"] in ("ok", "cached"))
        print(f"logos: {ok}/{len(logos_rows)} ok")
    if do_faces:
        print("---- FACES ----")
        faces_rows = fetch_faces()
        ok = sum(1 for r in faces_rows if r["status"] in ("ok", "cached"))
        miss = sum(1 for r in faces_rows if r["status"] == "no_hit")
        print(f"faces: {ok}/{len(faces_rows)} ok, {miss} no_hit")
    write_manifest(logos_rows, faces_rows)
    print(f"manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
